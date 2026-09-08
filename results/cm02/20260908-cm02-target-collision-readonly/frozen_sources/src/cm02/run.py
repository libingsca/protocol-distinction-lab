from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from cm01_rp.run import numpy_net, verify_assets
from .audit import audit
from .metrics import analyze

CONFIG = {
    'experiment': 'CM02', 'run_id': '20260908-cm02-target-collision-readonly',
    'splits': [9001, 9002, 9003], 'inits': [9201, 9202], 'arm': 'J-GROUP16',
    'epsilon_BCE': 1e-4, 'edge_tolerance': 1e-10, 'behavior_fit_min': .99,
    'primary': 'maximum near-optimal matching K_q < 16',
    'seconds_max': 1800, 'memory_bytes_max': 4 * 1024**3, 'threads': 1,
    'train_steps': 0, 'precision': 'float32 forward; stable float64 BCE and group means',
    'memory_enforcement': 'RSS high-water monitor every 0.1s; SIGUSR1 abort',
    'plan': 'docs/experiment-plans/cm02-target-collision-diagnostic.zh-CN.md',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def run(assets, output):
    assets, output = assets.resolve(), output.resolve()
    root = Path(__file__).resolve().parents[2]
    if output.is_relative_to(assets) or output.is_relative_to(root / 'results'):
        raise ValueError('Formal execution writes only to a fresh local output directory')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stop_monitor = threading.Event()

    def abort(signum, frame):
        raise RuntimeError('CM02 resource budget exceeded: ' + str(signum))

    signal.signal(signal.SIGALRM, abort)
    signal.signal(signal.SIGUSR1, abort)
    signal.alarm(CONFIG['seconds_max'])

    def monitor():
        while not stop_monitor.wait(.1):
            if rss() > CONFIG['memory_bytes_max']:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    threading.Thread(target=monitor, daemon=True).start()
    try:
        write(output / 'config.lock.json', CONFIG)
        sources = sorted((root / 'src/cm02').glob('*.py')) + [root / 'tests/test_cm02.py',
            root / 'src/cm01_rp/run.py', root / 'src/cm01_rp/metrics.py', root / CONFIG['plan']]
        source_hashes = {str(p.relative_to(root)): digest(p) for p in sources}
        write(output / 'source_hashes.json', source_hashes)
        for p in sources:
            dest = output / 'frozen_sources' / p.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
        (output / 'FROZEN_PLAN.zh-CN.md').write_bytes((root / CONFIG['plan']).read_bytes())
        write(output / 'environment.json', {'python': sys.version, 'numpy': np.__version__,
            'torch': torch.__version__, 'platform': platform.platform(), 'command': sys.argv,
            'dependencies': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
            'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'git_status_before': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
            'identity_note': 'Working tree has uncommitted CM02 files; source hashes define execution identity'})
        archive = root / 'evidence_readonly/joint02_minimal_assets/20260908/后续诊断最小实验资产.zip'
        assert digest(archive) == 'f7f6c249f40edf82e61aca68be30091f8a57ca0a6d0c0bf07cf8f13028c3849c'
        write(output / 'asset_verification.json', verify_assets(assets))
        # Hash all existing evidence before execution; no evaluation files are parsed.
        evidence = sorted(p for folder in ('results', 'evidence_readonly') for p in (root / folder).rglob('*') if p.is_file())
        evidence_hashes = {str(p.relative_to(root)): digest(p) for p in evidence}
        write(output / 'existing_evidence_hashes.json', evidence_hashes)
        allowed = [assets / 'frozen_sources/core.py', assets / 'frozen_sources/group_targets2.py']
        for split in CONFIG['splits']:
            allowed.append(assets / f'training_data/train_{split}.npz')
            for init in CONFIG['inits']:
                allowed.append(assets / f'models/{split}_{init}_J-GROUP16/checkpoint.pt')
        inputs = {str(p.relative_to(assets)): digest(p) for p in allowed}
        write(output / 'input_hashes.json', inputs)
        allowed_names = {str(p.resolve()) for p in allowed}
        accesses = set()
        restricted = True

        def guard(event, args):
            if restricted and event == 'open' and isinstance(args[0], (str, bytes)):
                p = Path(os.fsdecode(args[0])).resolve()
                if p.is_relative_to(assets):
                    if str(p) not in allowed_names:
                        raise PermissionError(str(p))
                    accesses.add(str(p.relative_to(assets)))

        sys.addaudithook(guard)
        denied = []
        for name in ['evaluation_only/evaluation_only.npz', 'historical_run/metrics.json']:
            try:
                (assets / name).read_bytes()
            except PermissionError:
                denied.append(name)
        assert len(denied) == 2
        sys.dont_write_bytecode = True
        loaded = {}
        for name in ('core', 'group_targets2'):
            path = assets / f'frozen_sources/{name}.py'
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
            loaded[name] = module
        core, grouping = loaded['core'], loaded['group_targets2']
        core.setup()
        qb, rb, mb = np.meshgrid(np.arange(2), np.arange(16), np.arange(16), indexing='ij')
        qflat = qb.ravel()
        rbits = ((rb.ravel()[:, None] >> np.arange(4)) & 1).astype(np.float32)
        mbits = np.concatenate([((mb.ravel()[:, None] >> np.arange(4)) & 1), np.zeros((512, 4))], axis=1).astype(np.float32)
        mask = np.tile([1, 1, 1, 1, 0, 0, 0, 0], (512, 1)).astype(np.float32)
        features = np.concatenate([mbits, mask, np.eye(2, dtype=np.float32)[qflat], rbits], axis=1)
        reports = []
        with torch.no_grad():
            for split in CONFIG['splits']:
                with np.load(assets / f'training_data/train_{split}.npz', allow_pickle=False) as f:
                    assert set(f.files) == {'x', 'q', 'r', 'y'}
                    train = {k: f[k].copy() for k in f.files}
                assert train['x'].shape == (4096, 8)
                assert train['r'].shape == train['y'].shape == (4096, 4)
                assert all(np.isin(a, [0, 1]).all() for a in train.values())
                groups = grouping.group_indices(train['x'], train['q'], train['r'])
                assert groups.shape == (256, 16)
                group_q = train['q'][groups[:, 0]].astype(int)
                group_x = train['x'][groups[:, 0]].astype(np.float32)
                group_y = train['y'][groups].astype(int)
                row_v = ((train['y'] * 2**np.arange(4)).sum(axis=1) - (train['r'] * 2**np.arange(4)).sum(axis=1)).astype(int) % 16
                assert np.all(row_v[groups] == row_v[groups[:, 0], None])
                group_v = row_v[groups[:, 0]]
                for q in range(2):
                    assert set(group_v[group_q == q]) == set(range(16))
                for init in CONFIG['inits']:
                    tag = f'{split}_{init}_J-GROUP16'
                    state = torch.load(assets / f'models/{tag}/checkpoint.pt', map_location='cpu', weights_only=True)
                    receiver, sender = core.Receiver().eval(), core.Sender().eval()
                    receiver.load_state_dict(state['receiver'])
                    sender.load_state_dict(state['sender'])
                    before = (core.state_hash(receiver.state_dict()), core.state_hash(sender.state_dict()))
                    raw = receiver(torch.from_numpy(mbits), torch.from_numpy(qflat), torch.from_numpy(rbits)).numpy()
                    independent = numpy_net(state['receiver'], features)
                    forward_error = float(np.max(np.abs(raw - independent)))
                    assert forward_error <= 1e-4 and np.array_equal(raw >= 0, independent >= 0)
                    logits = raw.reshape(2, 16, 16, 4)
                    messages = sender.message(torch.from_numpy(group_x), torch.from_numpy(group_q)).numpy()
                    si = numpy_net(state['sender'], np.concatenate([group_x, np.eye(2, dtype=np.float32)[group_q]], axis=1))[:, :4]
                    assert np.array_equal(messages[:, :4], si >= 0)
                    ids = (messages[:, :4] * 2**np.arange(4)).sum(axis=1).astype(int)
                    z = logits[group_q].astype(np.float64)
                    group_loss = (np.maximum(z, 0) - z * group_y[:, :, None, :] + np.log1p(np.exp(-np.abs(z)))).mean(axis=(1, 3))
                    group_accuracy = ((z >= 0) == group_y[:, :, None, :]).all(axis=3).mean(axis=1)
                    loss, accuracy = np.empty((2, 16, 16)), np.empty((2, 16, 16))
                    counts = np.zeros((2, 16, 16), dtype=int)
                    np.add.at(counts, (group_q, group_v, ids), 1)
                    for q in range(2):
                        for v in range(16):
                            ix = np.flatnonzero((group_q == q) & (group_v == v))
                            assert np.array_equal(group_loss[ix], np.repeat(group_loss[ix[:1]], len(ix), axis=0))
                            loss[q, v], accuracy[q, v] = group_loss[ix[0]], group_accuracy[ix[0]]
                    # Historical target operator receives original float32 bit losses.
                    row_q = train['q'].astype(int)
                    row_r = (train['r'] * 2**np.arange(4)).sum(axis=1).astype(int)
                    row_logits = torch.from_numpy(logits[row_q, row_r])
                    row_y = torch.from_numpy(train['y'].astype(np.float32))[:, None, :].expand(-1, 16, -1)
                    old_losses = torch.nn.functional.binary_cross_entropy_with_logits(row_logits, row_y, reduction='none').mean(dim=2)
                    _, old_targets = grouping.hard_targets(old_losses, groups)
                    historical_target = old_targets.numpy()[groups[:, 0]]
                    old_means = old_losses.numpy()[groups].astype(np.float64).mean(axis=1)
                    bce_error = float(np.max(np.abs(old_means - group_loss)))
                    assert bce_error <= 1e-6
                    conditions = []
                    for q in range(2):
                        condition = analyze(loss[q], accuracy[q], counts[q])
                        subset = group_q == q
                        condition.update(q=q, historical_target_fit=float(np.mean(ids[subset] == historical_target[subset])),
                            stable_vs_historical_target_disagreements=int(np.sum(group_loss[subset].argmin(axis=1) != historical_target[subset])))
                        conditions.append(condition)
                    assert before == (core.state_hash(receiver.state_dict()), core.state_hash(sender.state_dict()))
                    report = {'model': tag, 'conditions': conditions, 'forward_logits_max_error': forward_error,
                              'historical_BCE_max_error': bce_error, 'parameters_unchanged': True}
                    write(output / f'{tag}.json', report)
                    np.savez_compressed(output / f'{tag}_tables.npz', logits=logits,
                        numpy_logits=independent.reshape(2, 16, 16, 4), loss=loss, accuracy=accuracy,
                        counts=counts, group_x=group_x, group_q=group_q, group_v=group_v, group_y=group_y,
                        group_message=ids, historical_target=historical_target, historical_group_BCE=old_means)
                    reports.append(report)
                    print(tag, [(c['K'], c['F'], c['delta_star']) for c in conditions], flush=True)
        assert inputs == {name: digest(assets / name) for name in inputs}
        write(output / 'access_audit.json', {'passed': True, 'reads': sorted(accesses), 'denied_probes': denied,
            'input_hashes_unchanged': True, 'parameters_unchanged': True, 'evaluation_labels_parsed': False})
        restricted = False
        audit(output)
        assert evidence_hashes == {name: digest(root / name) for name in evidence_hashes}
        assert source_hashes == {name: digest(root / name) for name in source_hashes}
        elapsed = time.monotonic() - started
        assert elapsed <= CONFIG['seconds_max'] and rss() <= CONFIG['memory_bytes_max']
        write(output / 'integrity_audit.json', {'passed': True, 'existing_evidence_files_checked': len(evidence_hashes),
              'existing_evidence_unchanged': True, 'sources_unchanged': True, 'elapsed_seconds_including_audit': elapsed,
              'peak_rss_bytes': rss(), 'within_budget': True})
        write(output / 'summary.json', {'completed': True, 'audited': True, 'models': reports,
            'positive_conditions': sum(c['positive'] for r in reports for c in r['conditions']),
            'behavior_consistent_conditions': sum(c['behavior_consistent'] for r in reports for c in r['conditions']),
            'note': '12 conditions are paired within 6 historical models, not independent replications'})
    except BaseException as exc:
        write(output / 'FAILURE.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
              'elapsed_seconds': time.monotonic() - started, 'peak_rss_bytes': rss(),
              'scientific_conclusion': 'Execution incomplete; no scientific negative or positive declared'})
        raise
    finally:
        signal.alarm(0)
        stop_monitor.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.assets, args.output)

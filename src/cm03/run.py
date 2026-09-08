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

from .audit import audit
from .protocols import BRANCHES, freeze_model, validate_training
from .training import ADAM, compare, message_tensor, state_hash, train_branch


CONFIG = {
    'experiment': 'CM03', 'run_id': '20260908-cm03-injective-adaptation',
    'splits': [9001, 9002, 9003], 'inits': [9201, 9202], 'branches': BRANCHES,
    'steps': 100, 'batch_rows': 4096, 'optimizer': ADAM, 'threads': 1,
    'seconds_max': 3600, 'memory_bytes_max': 4 * 1024**3,
    'memory_enforcement': 'RSS high-water sampling every 0.1s; abort via SIGUSR1',
    'primary': {'models_min': 5, 'C0_max': -.01, 'C100_min': .01, 'G100_min': .05},
    'secondary': {'models_min': 5, 'selection_advantage_min': .01},
    'candidate_max_attempts': 10000, 'random_seed': '73001+100*model_index+10*q+k',
    'plan': 'docs/experiment-plans/cm03-injective-adaptation.zh-CN.md',
    'scope': 'Oracle training-label injective code; receiver-only adaptation; no held-out evaluation',
    'audit_tolerance': {'logits': 1e-4, 'BCE': 1e-6, 'decisions': 'exact'},
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def run(assets, cm02, output):
    root = Path(__file__).resolve().parents[2]
    assets, cm02, output = assets.resolve(), cm02.resolve(), output.resolve()
    assert output.is_relative_to(root / 'outputs/cm03')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stop_monitor = threading.Event()

    def abort(signum, frame):
        raise RuntimeError(f'CM03 resource budget exceeded: signal {signum}')

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
        assert torch.__version__ == '2.2.2' and np.__version__ == '1.26.4'
        sources = sorted((root / 'src/cm03').glob('*.py')) + [root / 'tests/test_cm03.py', root / CONFIG['plan']]
        source_hashes = {str(p.relative_to(root)): digest(p) for p in sources}
        write(output / 'source_hashes.json', source_hashes)
        for p in sources:
            dest = output / 'frozen_sources' / p.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
        (output / 'FROZEN_PLAN.zh-CN.md').write_bytes((root / CONFIG['plan']).read_bytes())
        write(output / 'environment.json', {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            'platform': platform.platform(), 'command': sys.argv,
            'dependencies': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
            'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
            'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'git_status_before': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
            'identity_note': 'Uncommitted working tree; frozen source hashes are execution identity'})
        existing = sorted(p for folder in ('results', 'evidence_readonly') for p in (root / folder).rglob('*') if p.is_file())
        evidence_hashes = {str(p.relative_to(root)): digest(p) for p in existing}
        write(output / 'existing_evidence_hashes.json', evidence_hashes)
        archive = root / 'evidence_readonly/joint02_minimal_assets/20260908/后续诊断最小实验资产.zip'
        assert digest(archive) == 'f7f6c249f40edf82e61aca68be30091f8a57ca0a6d0c0bf07cf8f13028c3849c'
        manifest = json.loads((assets / 'PACKAGE_MANIFEST.json').read_text())['files']
        for name, meta in manifest.items():
            assert (assets / name).stat().st_size == meta['bytes'] and digest(assets / name) == meta['sha256']
        for line in (cm02 / 'SHA256SUMS').read_text().splitlines():
            value, name = line.split('  ', 1)
            assert digest(cm02 / name) == value
        write(output / 'asset_verification.json', {'passed': True, 'asset_files_checked': len(manifest),
              'cm02_files_checked': len((cm02 / 'SHA256SUMS').read_text().splitlines()), 'archive_sha256': digest(archive)})
        tags = [f'{split}_{init}_J-GROUP16' for split in CONFIG['splits'] for init in CONFIG['inits']]
        allowed = [assets / 'frozen_sources/core.py']
        allowed += [assets / f'training_data/train_{split}.npz' for split in CONFIG['splits']]
        allowed += [assets / f'models/{tag}/checkpoint.pt' for tag in tags]
        allowed += [cm02 / f'{tag}.json' for tag in tags]
        input_hashes = {str(p): digest(p) for p in allowed}
        write(output / 'input_hashes.json', input_hashes)
        (output / 'frozen_sources/historical_core.py').write_bytes((assets / 'frozen_sources/core.py').read_bytes())
        allowed_names, accesses = {str(p) for p in allowed}, set()
        restricted = True

        def guard(event, args):
            if restricted and event == 'open' and isinstance(args[0], (str, bytes)):
                path = Path(os.fsdecode(args[0])).resolve()
                if path.is_relative_to(assets) or path.is_relative_to(root / 'results') or path.is_relative_to(root / 'evidence_readonly'):
                    if str(path) not in allowed_names:
                        raise PermissionError(str(path))
                    accesses.add(str(path))

        sys.addaudithook(guard)
        denied = []
        for path in (assets / 'evaluation_only/evaluation_only.npz', assets / 'historical_run/metrics.json', cm02 / 'summary.json'):
            try:
                path.read_bytes()
            except PermissionError:
                denied.append(str(path))
        assert len(denied) == 3
        sys.dont_write_bytecode = True
        core_path = assets / 'frozen_sources/core.py'
        spec = importlib.util.spec_from_file_location('cm03_frozen_core', core_path)
        core = importlib.util.module_from_spec(spec)
        exec(compile(core_path.read_text(), str(core_path), 'exec'), core.__dict__)
        core.setup()
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
        train_by_split, v_by_split, first_by_split = {}, {}, {}
        for split in CONFIG['splits']:
            with np.load(assets / f'training_data/train_{split}.npz', allow_pickle=False) as data:
                train = {key: data[key].copy() for key in data.files}
            v, first = validate_training(train)
            train_by_split[split], v_by_split[split], first_by_split[split] = train, v, first
        # Freeze every model's code table before the first receiver update.
        protocols, row_tables, origins = {}, {}, {}
        with torch.no_grad():
            for i, tag in enumerate(tags):
                split = int(tag.split('_')[0])
                train = train_by_split[split]
                state = torch.load(assets / f'models/{tag}/checkpoint.pt', map_location='cpu', weights_only=True)
                sender = core.Sender().eval()
                sender.load_state_dict(state['sender'])
                sender_before = state_hash(sender.state_dict())
                message = sender.message(torch.from_numpy(train['x'].astype(np.float32)), torch.from_numpy(train['q'].astype(np.int64))).numpy()
                original = (message[:, :4] * 2**np.arange(4)).sum(axis=1).astype(int)
                assert state_hash(sender.state_dict()) == sender_before
                source = json.loads((cm02 / f'{tag}.json').read_text())
                target_map = [condition['minimum_cost_assignment'] for condition in source['conditions']]
                protocol, rows = freeze_model(i, train['q'].astype(int), v_by_split[split], first_by_split[split], original, target_map)
                protocol['original_sender_hash'] = sender_before
                protocol['receiver_initial_hash'] = state_hash(state['receiver'])
                protocol['cm02_assignment_source_hash'] = input_hashes[str(cm02 / f'{tag}.json')]
                protocols[tag], row_tables[tag], origins[tag] = protocol, rows, state
                np.savez_compressed(output / f'{tag}_protocol_rows.npz', **rows)
        write(output / 'protocols.lock.json', protocols)
        protocol_files = [output / 'protocols.lock.json'] + [output / f'{tag}_protocol_rows.npz' for tag in tags]
        protocol_hashes = {p.name: digest(p) for p in protocol_files}
        write(output / 'protocol_hashes.json', protocol_hashes)
        print('All six candidate sets frozen; beginning 3000 full-batch updates.', flush=True)
        reports = []
        for tag in tags:
            split = int(tag.split('_')[0])
            train = train_by_split[split]
            q = torch.from_numpy(train['q'].astype(np.int64))
            r, y = [torch.from_numpy(train[key].astype(np.float32)) for key in ('r', 'y')]
            branches = {}
            for name in BRANCHES:
                receiver = core.Receiver()
                branch, endpoints = train_branch(receiver, origins[tag]['receiver'], message_tensor(row_tables[tag][name]),
                    q, r, y, CONFIG['steps'], output / f'{tag}_{name}_weights.pt')
                branches[name] = branch
                np.savez_compressed(output / f'{tag}_{name}_endpoints.npz', logits=endpoints)
                write(output / f'{tag}_{name}_curve.json', branch)
                print(tag, name, 'BCE', branch['curve'][0]['BCE'], '->', branch['curve'][-1]['BCE'],
                      'accuracy', branch['curve'][-1]['accuracy'], flush=True)
            assert len({branch['initial_hash'] for branch in branches.values()}) == 1
            assert state_hash(origins[tag]['receiver']) == protocols[tag]['receiver_initial_hash']
            report = {'model': tag, 'branches': branches, 'result': compare(branches)}
            write(output / f'{tag}.json', report)
            reports.append({'model': tag, **report['result']})
        passes = sum(r['target_passes'] for r in reports)
        advantage_passes = sum(r['selection_advantage_passes'] for r in reports)
        summary = {'models': reports, 'primary_pass_count': passes, 'primary_passed': passes >= 5,
                   'selection_advantage_count': advantage_passes, 'selection_advantage_gate_passed': advantage_passes >= 5,
                   'branches_completed': 30, 'full_batch_updates': 3000,
                   'scope': 'Six historical models; not independent statistical replicates or held-out generalization'}
        write(output / 'summary.pending.json', summary)
        print('Training complete; independently auditing candidates, 60 endpoints and all gates.', flush=True)
        audit(output, assets, cm02)
        assert input_hashes == {name: digest(name) for name in input_hashes}
        assert protocol_hashes == {name: digest(output / name) for name in protocol_hashes}
        write(output / 'access_audit.json', {'passed': True, 'reads': sorted(accesses), 'denied_probes': denied,
              'input_hashes_unchanged': True, 'protocol_hashes_unchanged': True, 'evaluation_labels_parsed': False})
        restricted = False
        assert evidence_hashes == {name: digest(root / name) for name in evidence_hashes}
        assert source_hashes == {name: digest(root / name) for name in source_hashes}
        elapsed = time.monotonic() - started
        assert elapsed <= CONFIG['seconds_max'] and rss() <= CONFIG['memory_bytes_max']
        write(output / 'integrity_audit.json', {'passed': True, 'existing_evidence_files_checked': len(evidence_hashes),
              'existing_evidence_unchanged': True, 'sources_unchanged': True, 'elapsed_seconds_including_audit': elapsed,
              'peak_rss_bytes': rss(), 'within_budget': True})
        write(output / 'summary.json', {'completed': True, 'audited': True, **summary})
        print('CM03 audited: primary', passes, '/6; selection advantage', advantage_passes, '/6', flush=True)
    except BaseException as exc:
        write(output / 'FAILURE.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
              'elapsed_seconds': time.monotonic() - started, 'peak_rss_bytes': rss(),
              'scientific_conclusion': 'Execution incomplete; no scientific conclusion declared'})
        raise
    finally:
        signal.alarm(0)
        stop_monitor.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--cm02', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.assets, args.cm02, args.output)

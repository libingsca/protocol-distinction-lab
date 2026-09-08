from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .metrics import diagnose

CONFIG = {
    'experiment': 'CM01-RP', 'version': '1.0', 'splits': [9001, 9002, 9003],
    'inits': [9201, 9202], 'arm': 'J-GROUP16', 'checkpoint': 'checkpoint.pt',
    'mean_tv_max': .05, 'max_tv_max': .10, 'coverage_required_r': 16,
    'logits_audit_atol': 1e-4, 'q': [0, 1], 'r': list(range(16)), 'messages': list(range(16)),
    'scope': 'read-only structural diagnosis; no adaptation or candidate selection',
    'precision': 'PyTorch float32 inference; NumPy float64 probability distances',
    'positive': 'at least one q with occupied near pair AND missing necessary V',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify_assets(root):
    manifest = json.loads((root / 'PACKAGE_MANIFEST.json').read_text())
    failures = [name for name, meta in manifest['files'].items()
                if not (root/name).is_file() or (root/name).stat().st_size != meta['bytes']
                or digest(root/name) != meta['sha256']]
    if failures:
        raise ValueError(f'Asset checksum failures: {failures}')
    return {'checked': len(manifest['files']), 'passed': True,
            'manifest_sha256': digest(root/'PACKAGE_MANIFEST.json')}


def numpy_net(state, x):
    for layer in (0, 2, 4):
        x = x @ state[f'net.{layer}.weight'].numpy().T + state[f'net.{layer}.bias'].numpy()
        if layer != 4:
            x = np.maximum(x, 0)
    return x


def source_hashes(root):
    files = sorted((root/'src/cm01_rp').glob('*.py')) + sorted((root/'tests').glob('test_cm01_rp.py'))
    files += [root/'docs/experiment-plans/cm01-rp-frozen-20260908.zh-CN.md']
    return {str(p.relative_to(root)): digest(p) for p in files}


def run(assets, output):
    assets, output = assets.resolve(), output.resolve()
    if output.is_relative_to(assets):
        raise ValueError('Output must be outside assets')
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = source_hashes(root)
    write(output/'config.lock.json', CONFIG)
    write(output/'source_hashes.json', sources)
    write(output/'environment.json', {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
          'platform': platform.platform(), 'started_at': datetime.now(timezone.utc).isoformat(),
          'command': sys.argv, 'threads': 1,
          'dependencies': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)})
    write(output/'asset_verification.json', verify_assets(assets))
    allowed = [assets/'frozen_sources/core.py']
    for split in CONFIG['splits']:
        allowed.append(assets/f'training_data/train_{split}.npz')
        for init in CONFIG['inits']:
            allowed.append(assets/f'models/{split}_{init}_J-GROUP16/checkpoint.pt')
    inputs = {str(p.relative_to(assets)): digest(p) for p in allowed}
    write(output/'input_hashes.json', inputs)
    allowed = {str(p.resolve()) for p in allowed}
    access = set()
    guard_enabled = True
    def guard(event, args):
        if guard_enabled and event == 'open' and isinstance(args[0], (str, bytes)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
            if path.is_relative_to(assets):
                if str(path) not in allowed:
                    raise PermissionError(f'Asset access outside diagnostic allowlist: {path}')
                access.add(str(path.relative_to(assets)))
    sys.addaudithook(guard)
    denied = []
    for relative in ['evaluation_only/evaluation_only.npz', 'historical_run/metrics.json']:
        try:
            (assets/relative).read_bytes()
        except PermissionError:
            denied.append(relative)
    assert len(denied) == 2
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('cm01_frozen_core', assets/'frozen_sources/core.py')
    core = importlib.util.module_from_spec(spec)
    # Compile verified source directly to avoid an implicit bytecode-cache read.
    exec(compile((assets/'frozen_sources/core.py').read_text(), str(assets/'frozen_sources/core.py'), 'exec'), core.__dict__)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    qb, rb, mb = np.meshgrid(np.arange(2), np.arange(16), np.arange(16), indexing='ij')
    q = qb.ravel()
    r = ((rb.ravel()[:, None] >> np.arange(4)) & 1).astype(np.float32)
    m = np.concatenate([((mb.ravel()[:, None] >> np.arange(4)) & 1), np.zeros((512,4))], axis=1).astype(np.float32)
    mask = np.tile(np.array([1,1,1,1,0,0,0,0], dtype=np.float32), (512,1))
    receiver_input = np.concatenate([m, mask, np.eye(2, dtype=np.float32)[q], r], axis=1)
    reports = []
    with torch.no_grad():
        for split in CONFIG['splits']:
            with np.load(assets/f'training_data/train_{split}.npz', allow_pickle=False) as f:
                assert set(f.files) == {'x', 'q', 'r', 'y'}
                train = {k: f[k].copy() for k in f.files}
            assert train['x'].shape == (4096,8) and train['r'].shape == train['y'].shape == (4096,4)
            for k in ('x','q','r','y'):
                assert np.isin(train[k], [0,1]).all()
            train_q = train['q'].astype(int)
            v = ((train['y'] * (2**np.arange(4))).sum(1) - (train['r'] * (2**np.arange(4))).sum(1)).astype(int) % 16
            required = [np.unique(v[train_q == qi]).tolist() for qi in range(2)]
            unique, ix, inverse, counts = np.unique(np.column_stack([train['x'],train_q]), axis=0, return_index=True, return_inverse=True, return_counts=True)
            assert len(ix) == 256 and np.all(counts == 16)
            for group in range(256):
                rows = np.flatnonzero(inverse == group)
                assert len(np.unique(v[rows])) == 1
                assert sorted((train['r'][rows] * (2**np.arange(4))).sum(1).tolist()) == list(range(16))
            for init in CONFIG['inits']:
                tag = f'{split}_{init}_J-GROUP16'
                state = torch.load(assets/f'models/{tag}/checkpoint.pt', map_location='cpu', weights_only=True)
                receiver, sender = core.Receiver().eval(), core.Sender().eval()
                receiver.load_state_dict(state['receiver']); sender.load_state_dict(state['sender'])
                before = (core.state_hash(receiver.state_dict()), core.state_hash(sender.state_dict()))
                logits = receiver(torch.from_numpy(m),torch.from_numpy(q),torch.from_numpy(r)).numpy()
                independent = numpy_net(state['receiver'], receiver_input)
                error = float(np.max(np.abs(logits-independent)))
                assert error <= CONFIG['logits_audit_atol'] and np.array_equal(logits>=0,independent>=0)
                x = train['x'][ix].astype(np.float32); tq = train_q[ix]
                message = sender.message(torch.from_numpy(x), torch.from_numpy(tq)).numpy()
                independent_sender = numpy_net(state['sender'], np.concatenate([x,np.eye(2,dtype=np.float32)[tq]],axis=1))[:,:4]
                assert np.array_equal(message[:,:4], independent_sender>=0)
                ids = (message[:,:4] * (2**np.arange(4))).sum(1).astype(int)
                usage = np.zeros((2,16),dtype=int)
                np.add.at(usage,(tq,ids),1)
                report, arrays = diagnose(logits.reshape(2,16,16,4), usage, required,
                                          CONFIG['mean_tv_max'], CONFIG['max_tv_max'])
                assert before == (core.state_hash(receiver.state_dict()), core.state_hash(sender.state_dict()))
                report.update(model=tag, split=split, init=init, independent_logits_max_error=error,
                              independent_decisions_equal=True, parameters_unchanged=True)
                np.savez_compressed(output/f'{tag}_responses.npz', logits=logits.reshape(2,16,16,4),
                                    usage=usage, train_group_messages=ids, train_group_q=tq, **arrays)
                write(output/f'{tag}.json',report)
                reports.append(report)
                print(tag, 'positive=', report['positive'], flush=True)
    assert inputs == {name:digest(assets/name) for name in inputs}
    assert sources == source_hashes(root)
    write(output/'access_audit.json', {'passed':True, 'inference_asset_reads':sorted(access),
        'denied_probes':denied, 'parameters_unchanged':True, 'input_hashes_unchanged':True,
        'source_hashes_unchanged':True, 'note':'Full-package hashing happened before restricted inference; no evaluation labels parsed.'})
    guard_enabled = False
    write(output/'summary.json', {'model_count':len(reports), 'positive_count':sum(r['positive'] for r in reports),
        'eligible_models':[r['model'] for r in reports if r['positive']], 'models':reports,
        'completed_at':datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.assets,args.output)

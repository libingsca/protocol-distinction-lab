import argparse
import hashlib
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
from .core import (ADAM, ENCODINGS, REPRESENTATIONS, Receiver, compare_seed, encodings,
                   features, paired_state, state_hash, summarize, symbolic_check, table, train)


CONFIG = {'experiment': 'CM04', 'run_id': '20260908-cm04-code-geometry',
    'seeds': list(range(60, 90)), 'permutation_seeds': [84001, 84002, 84003],
    'encodings': ENCODINGS, 'representations': REPRESENTATIONS, 'shape': [22, 64, 64, 4],
    'train_rows': 512, 'steps': 100, 'optimizer': ADAM, 'threads': 1,
    'primary': {'per_mapping_joint_seeds_min': 24, 'D_loss_min': .01, 'D_acc_min': .05},
    'control': {'BCE_diff_max': 1e-5, 'accuracy_diff_max': 1e-5, 'endpoint_logits_diff_max': 1e-4},
    'audit': {'BCE_tolerance': 1e-6, 'logits_tolerance': 1e-4, 'decisions': 'exact'},
    'seconds_max': 3600, 'rss_bytes_max': 4 * 1024**3,
    'memory_enforcement': 'RSS high-water monitor every 0.1s; SIGUSR1 abort',
    'plan': 'docs/experiment-plans/cm04-code-geometry.zh-CN.md',
    'scope': 'Oracle balanced training table; fresh initialization; explicitly paired one-hot parameters'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def run(output):
    root = Path(__file__).resolve().parents[2]
    output = output.resolve()
    assert output.is_relative_to(root / 'outputs/cm04')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stop = threading.Event()

    def abort(signum, frame):
        raise RuntimeError(f'CM04 resource budget exceeded; signal {signum}')

    signal.signal(signal.SIGALRM, abort)
    signal.signal(signal.SIGUSR1, abort)
    signal.alarm(CONFIG['seconds_max'])

    def monitor():
        while not stop.wait(.1):
            if rss() > CONFIG['rss_bytes_max']:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    threading.Thread(target=monitor, daemon=True).start()
    try:
        assert torch.__version__ == '2.2.2' and np.__version__ == '1.26.4'
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        write(output / 'config.lock.json', CONFIG)
        sources = sorted((root/'src/cm04').glob('*.py')) + [root/'tests/test_cm04.py', root/CONFIG['plan']]
        source_hashes = {str(p.relative_to(root)): digest(p) for p in sources}
        write(output/'source_hashes.json', source_hashes)
        for p in sources:
            dest = output/'frozen_sources'/p.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
        (output/'FROZEN_PLAN.zh-CN.md').write_bytes((root/CONFIG['plan']).read_bytes())
        write(output/'environment.json', {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            'platform': platform.platform(), 'command': sys.argv, 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
            'dependencies': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
            'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
            'identity_note': 'Uncommitted source; source hashes define execution identity'})
        evidence = sorted(p for name in ('results', 'evidence_readonly') for p in (root/name).rglob('*') if p.is_file())
        evidence_hashes = {str(p.relative_to(root)): digest(p) for p in evidence}
        write(output/'existing_evidence_hashes.json', evidence_hashes)
        protected = [root/'results', root/'evidence_readonly', root/'outputs/cm01_rp/assets-20260908']
        restricted = True

        def guard(event, args):
            if restricted and event == 'open' and isinstance(args[0], (str, bytes)):
                path = Path(os.fsdecode(args[0])).resolve()
                if any(path.is_relative_to(folder) for folder in protected):
                    raise PermissionError(f'Historical input prohibited for CM04: {path}')

        sys.addaudithook(guard)
        denied = []
        for path in (root/'results/cm03/20260908-cm03-injective-adaptation/summary.json',
                     root/'outputs/cm01_rp/assets-20260908/02_实验资产/evaluation_only/evaluation_only.npz',
                     root/'outputs/cm01_rp/assets-20260908/02_实验资产/models/9001_9201_J-GROUP16/checkpoint.pt'):
            try:
                path.read_bytes()
            except PermissionError:
                denied.append(str(path))
        assert len(denied) == 3
        data, maps = table(), encodings()
        symbolic = symbolic_check(data, maps)
        write(output/'symbolic_checks.json', symbolic)
        write(output/'encodings.lock.json', {'mappings': maps, 'seeds': CONFIG['permutation_seeds'],
            'duplicate_pairs': [[a, b] for i, a in enumerate(ENCODINGS) for b in ENCODINGS[i+1:] if maps[a] == maps[b]],
            'fixed_points': {name: sum(i == m for i, m in enumerate(mapping)) for name, mapping in maps.items()}})
        np.savez_compressed(output/'training_table.npz', **data)
        inputs = {f'{rep}_{name}': features(data, maps[name], rep) for rep in REPRESENTATIONS for name in ENCODINGS}
        np.savez_compressed(output/'inputs.lock.npz', **inputs)
        initial_states, initial_hashes = {}, {}
        for seed in CONFIG['seeds']:
            torch.manual_seed(seed)
            model = Receiver()
            state = {key: value.clone() for key, value in model.state_dict().items()}
            initial_states[seed], initial_hashes[str(seed)] = state, state_hash(state)
            torch.save(state, output/f'initial_{seed}.pt')
        write(output/'initialization.lock.json', initial_hashes)
        frozen_inputs = [output/'encodings.lock.json', output/'training_table.npz', output/'inputs.lock.npz', output/'initialization.lock.json']
        frozen_inputs += [output/f'initial_{seed}.pt' for seed in CONFIG['seeds']]
        input_hashes = {p.name: digest(p) for p in frozen_inputs}
        write(output/'input_hashes.json', input_hashes)
        print('Four encodings, 512 rows and 30 initializations frozen before first update.', flush=True)
        reports = []
        q, y = torch.from_numpy(data['q']), torch.from_numpy(data['y'])
        for seed in CONFIG['seeds']:
            branches, endpoints = {}, {}
            for rep in REPRESENTATIONS:
                for name in ENCODINGS:
                    key = f'{rep}_{name}'
                    initial = paired_state(initial_states[seed], maps[name], rep)
                    branch, logits = train(initial, torch.from_numpy(inputs[key]), q, y, CONFIG['steps'],
                                           output/f'{seed}_{key}_weights.pt')
                    branches[key], endpoints[key] = branch, logits
                    write(output/f'{seed}_{key}_curve.json', branch)
                    np.savez_compressed(output/f'{seed}_{key}_endpoints.npz', logits=logits)
            result = compare_seed(branches, endpoints)
            report = {'seed': seed, 'branches': branches, 'result': result}
            write(output/f'seed_{seed}.json', report)
            reports.append(report)
            assert state_hash(initial_states[seed]) == initial_hashes[str(seed)]
            print('seed', seed, 'bit pass', [result['bit_comparisons'][k]['joint_pass'] for k in ENCODINGS[1:]],
                  'one-hot control', result['control_passed'], flush=True)
        summary = summarize(reports)
        write(output/'summary.pending.json', summary)
        print('Training completed; auditing 480 endpoints and every control trajectory.', flush=True)
        audit(output)
        assert input_hashes == {name: digest(output/name) for name in input_hashes}
        write(output/'access_audit.json', {'passed': True, 'denied_probes': denied,
            'historical_checkpoint_loaded': False, 'historical_labels_parsed': False, 'input_hashes_unchanged': True,
            'note': 'Existing evidence was byte-hashed before protected execution; no historical contents parsed'})
        restricted = False
        assert evidence_hashes == {name: digest(root/name) for name in evidence_hashes}
        assert source_hashes == {name: digest(root/name) for name in source_hashes}
        elapsed = time.monotonic()-started
        assert elapsed <= CONFIG['seconds_max'] and rss() <= CONFIG['rss_bytes_max']
        write(output/'integrity_audit.json', {'passed': True, 'existing_evidence_files_checked': len(evidence_hashes),
            'existing_evidence_unchanged': True, 'sources_unchanged': True, 'elapsed_seconds_including_audit': elapsed,
            'peak_rss_bytes': rss(), 'within_budget': True})
        verdict = 'control_failure' if not summary['control_passed'] else ('primary_passed' if summary['primary_passed'] else 'primary_failed')
        write(output/'summary.json', {'completed': True, 'audited': True, 'verdict': verdict, **summary})
        if not summary['control_passed']:
            write(output/'CONTROL_FAILURE.json', {'verdict': 'No mechanism inference allowed',
                'failed_pairs': [{'seed': r['seed'], 'encoding': name, **c} for r in reports for name, c in r['result']['onehot_controls'].items() if not c['passed']]})
        print('Audited verdict:', verdict, summary, flush=True)
    except BaseException as exc:
        write(output/'FAILURE.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
            'elapsed_seconds': time.monotonic()-started, 'peak_rss_bytes': rss(),
            'scientific_conclusion': 'Execution/audit incomplete; no mechanism conclusion'})
        raise
    finally:
        signal.alarm(0)
        stop.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)

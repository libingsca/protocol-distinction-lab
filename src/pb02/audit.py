"""Recompute all gates from saved branch curves without rerunning training."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from .run import CONFIG, hashes, write_json


def audit(output):
    read = lambda name: json.loads((output / name).read_text())
    assert read('config.lock.json') == CONFIG
    assert read('source_hashes.json') == hashes(Path(__file__).resolve().parents[2])
    summary = read('summary.json')
    details = {}
    for architecture in CONFIG['architectures']:
        rows = [read(f'{architecture}_seed_{seed}.json') for seed in range(30, 60)]
        c0, c2, gains, reductions, random, noninfo = [], [], [], [], [], []
        for seed, row in zip(range(30, 60), rows):
            assert row['seed'] == seed and row['architecture'] == architecture
            assert row['target'] == CONFIG['target']
            branches = row['branches']
            c, d = branches['C'], branches['D']
            credits = [c[str(t)][0] - d[str(t)][0] for t in range(33)]
            delta = [d[str(t)][1] - c[str(t)][1] for t in range(33)]
            assert credits == row['credits'] and delta == row['accuracy_gains']
            assert row['c0_ce'] == credits[0] and row['c2_ce'] == credits[2]
            assert row['accuracy_gain_t4'] == delta[4]
            assert row['first_loss'] == next((t for t in range(33) if credits[t] > 0), None)
            assert row['first_decision'] == next((t for t in range(33) if delta[t] >= .1), None)
            c0.append(credits[0]); c2.append(credits[2]); gains.append(delta[4])
            reductions.append(1 - abs(branches['spare_C']['0'][0] - branches['spare_D']['0'][0]) / abs(credits[0]))
            random.append(c['2'][0] - branches['random']['2'][0])
            noninfo.append(c['2'][0] - branches['noninfo']['2'][0])
            assert reductions[-1] == row['spare_barrier_reduction']
            assert random[-1] == row['random_c2_ce'] and noninfo[-1] == row['noninfo_c2_ce']
        med = statistics.median
        gates = [sum(a < 0 < b for a, b in zip(c0, c2)) >= 24, med(c0) <= -.05,
                 med(c2) >= .02, sum(g >= .1 for g in gains) >= 24, med(gains) >= .1,
                 med(reductions) >= .5, med(c2) > med(random) and med(c2) > med(noninfo)]
        assert gates == list(summary['architectures'][architecture]['checks'].values())
        assert all(gates) == summary['architectures'][architecture]['passed']
        details[architecture] = {'row_count': len(rows), 'independent_gates': gates}
    assert summary['primary_passed'] == summary['architectures']['mlp8']['passed']
    write_json(output / 'audit.json', {'passed': True, 'details': details,
        'checks': ['locked configuration', 'source hashes unchanged', 'exact seed coverage',
                   'fixed targets', 'full branch arithmetic', 'first crossing times', 'seven independent gates']})
    lines = []
    for path in sorted(output.glob('*.json')):
        lines.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}')
    (output / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output)

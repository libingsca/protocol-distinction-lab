"""Independent candidate reconstruction, NumPy forward and scalar endpoint audit.

Does not import candidate construction, training or verdict implementation.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def hash_parameters(state):
    h = hashlib.sha256()
    for key in state:
        h.update(key.encode())
        h.update(state[key].cpu().numpy().tobytes())
    return h.hexdigest()


def numpy_forward(state, inputs):
    value = inputs.astype(np.float64)
    for layer in (0, 2, 4):
        value = value @ state[f'net.{layer}.weight'].numpy().astype(np.float64).T
        value += state[f'net.{layer}.bias'].numpy().astype(np.float64)
        if layer != 4:
            value = np.maximum(value, 0.)
    return value


def scalar_metrics(logits, labels, q):
    loss_by_q, correct_by_q = {0: [], 1: []}, {0: [], 1: []}
    for zrow, yrow, qi in zip(logits, labels, q):
        terms, correct = [], True
        for z, y in zip(zrow, yrow):
            z, y = float(z), int(y)
            terms.append(max(z, 0.) - z*y + math.log1p(math.exp(-abs(z))))
            correct = correct and ((z >= 0) == bool(y))
        loss_by_q[int(qi)].append(math.fsum(terms) / 4)
        correct_by_q[int(qi)].append(int(correct))
    return {'BCE': math.fsum(loss_by_q[0] + loss_by_q[1]) / len(q),
            'accuracy': sum(correct_by_q[0] + correct_by_q[1]) / len(q),
            'by_q': [{'q': qi, 'BCE': math.fsum(loss_by_q[qi]) / len(loss_by_q[qi]),
                      'accuracy': sum(correct_by_q[qi]) / len(correct_by_q[qi])} for qi in (0, 1)]}


def check_metrics(observed, expected):
    assert abs(observed['BCE'] - expected['BCE']) <= 1e-6
    assert observed['accuracy'] == expected['accuracy']
    for qi in (0, 1):
        assert abs(observed['by_q'][qi]['BCE'] - expected['by_q'][qi]['BCE']) <= 1e-6
        assert observed['by_q'][qi]['accuracy'] == expected['by_q'][qi]['accuracy']


def audit(output, assets, cm02):
    output, assets, cm02 = Path(output), Path(assets), Path(cm02)
    locked = json.loads((output / 'protocols.lock.json').read_text())
    summary = json.loads((output / 'summary.pending.json').read_text())
    models, endpoint_audits = [], []
    for index, tag in enumerate(locked):
        split, init = tag.split('_')[:2]
        with np.load(assets / f'training_data/train_{split}.npz', allow_pickle=False) as data:
            train = {key: data[key] for key in data.files}
        with np.load(output / f'{tag}_protocol_rows.npz', allow_pickle=False) as data:
            row_messages = {key: data[key] for key in data.files}
        q = train['q'].astype(int)
        v, group_rows = [], {}
        for i in range(len(q)):
            rv = sum(int(train['r'][i, b]) << b for b in range(4))
            yv = sum(int(train['y'][i, b]) << b for b in range(4))
            v.append((yv - rv) % 16)
            key = tuple(train['x'][i].tolist()) + (int(q[i]),)
            group_rows.setdefault(key, []).append(i)
        v = np.asarray(v)
        assert len(group_rows) == 256
        for rows in group_rows.values():
            assert len(rows) == 16 and len(set(v[rows])) == 1
            assert sorted(sum(int(train['r'][i, b]) << b for b in range(4)) for i in rows) == list(range(16))
        origin = torch.load(assets / f'models/{tag}/checkpoint.pt', map_location='cpu', weights_only=True)
        sender_input = np.column_stack([train['x'], q == 0, q == 1])
        sender_logits = numpy_forward(origin['sender'], sender_input)[:, :4]
        original = (sender_logits >= 0).astype(int) @ np.array([1, 2, 4, 8])
        assert np.array_equal(original, row_messages['original'])
        old_map = np.empty((2, 16), dtype=int)
        for qi in (0, 1):
            assert set(v[q == qi]) == set(range(16))
            for vi in range(16):
                options = set(original[(q == qi) & (v == vi)].tolist())
                assert len(options) == 1
                old_map[qi, vi] = options.pop()
        cm02_result = json.loads((cm02 / f'{tag}.json').read_text())
        target = np.array([c['minimum_cost_assignment'] for c in cm02_result['conditions']])
        assert np.array_equal(old_map, locked[tag]['maps']['original'])
        assert np.array_equal(target, locked[tag]['maps']['target'])
        maps = {'original': old_map, 'target': target}
        for k in range(3):
            name = f'random{k+1}'
            maps[name] = target.copy()
            for qi in (0, 1):
                seed = 73001 + 100*index + 10*qi + k
                rng = np.random.Generator(np.random.PCG64(seed))
                changed = [vi for vi in range(16) if old_map[qi, vi] != target[qi, vi]]
                for attempt in range(1, 10001):
                    candidate = target[qi].copy()
                    candidate[changed] = rng.permutation(target[qi, changed])
                    if all(candidate[vi] != old_map[qi, vi] for vi in changed) and candidate.tolist() != target[qi].tolist():
                        break
                else:
                    raise AssertionError('Candidate cannot be reconstructed')
                record = locked[tag]['random_generation'][name][qi]
                assert record['seed'] == seed and record['attempts'] == attempt and record['changed_V'] == changed
                maps[name][qi] = candidate
            assert np.array_equal(maps[name], locked[tag]['maps'][name])
        repeated = [[a, b] for i, a in enumerate(('random1', 'random2', 'random3'))
                    for b in ('random1', 'random2', 'random3')[i+1:] if np.array_equal(maps[a], maps[b])]
        assert repeated == locked[tag]['duplicate_random_branches']
        model = json.loads((output / f'{tag}.json').read_text())
        scalar_endpoints = {}
        for name, mapping in maps.items():
            assert np.array_equal(mapping[q, v], row_messages[name])
            if name != 'original':
                for qi in (0, 1):
                    assert sorted(mapping[qi]) == list(range(16))
                assert np.array_equal(row_messages[name] != original, row_messages['target'] != original)
            for qi in (0, 1):
                groups = [rows[0] for key, rows in group_rows.items() if key[-1] == qi]
                changed_fraction = sum(row_messages[name][i] != original[i] for i in groups) / len(groups)
                bit_hamming = sum(int(row_messages[name][i] ^ original[i]).bit_count() for i in np.flatnonzero(q == qi)) / sum(q == qi)
                record = locked[tag]['distances'][name][qi]
                assert record['changed_group_fraction'] == changed_fraction
                assert record['mean_bit_hamming'] == bit_hamming
            checkpoint = torch.load(output / f'{tag}_{name}_weights.pt', map_location='cpu', weights_only=True)
            assert checkpoint['initial_optimizer']['state'] == {}
            assert all(torch.equal(value, checkpoint['initial_receiver'][key]) for key, value in origin['receiver'].items())
            for stage in ('initial', 'final'):
                settings = checkpoint[f'{stage}_optimizer']['param_groups']
                assert len(settings) == 1
                settings = settings[0]
                assert settings['lr'] == .001 and settings['betas'] == (.9, .999)
                assert settings['eps'] == 1e-8 and settings['weight_decay'] == 0
            assert len(checkpoint['final_optimizer']['state']) == len(origin['receiver'])
            assert all(int(s['step']) == 100 for s in checkpoint['final_optimizer']['state'].values())
            curve = model['branches'][name]['curve']
            assert [x['step'] for x in curve] == list(range(101))
            assert model['branches'][name]['batch_rows'] == 4096
            assert model['branches'][name]['steps'] == 100
            assert model['branches'][name]['optimizer_initial_state_count'] == 0
            assert curve[0]['parameter_hash'] == hash_parameters(origin['receiver'])
            for row in curve:
                assert len(row['parameter_hash']) == 64 and math.isfinite(row['BCE']) and row['BCE'] >= 0
                assert 0 <= row['accuracy'] <= 1
                assert abs(row['BCE'] - sum(row['by_q'][qi]['BCE'] * np.mean(q == qi) for qi in (0, 1))) <= 1e-6
                assert row['accuracy'] == sum(row['by_q'][qi]['accuracy'] * np.mean(q == qi) for qi in (0, 1))
            with np.load(output / f'{tag}_{name}_endpoints.npz', allow_pickle=False) as data:
                saved_logits = data['logits']
            assert saved_logits.shape == (2, 4096, 4)
            ids = row_messages[name]
            bits = np.array([[(int(m) >> b) & 1 for b in range(4)] for m in ids])
            features = np.column_stack([bits, np.zeros((4096, 4)), np.tile([1, 1, 1, 1, 0, 0, 0, 0], (4096, 1)), q == 0, q == 1, train['r']])
            scalar_endpoints[name] = []
            for endpoint, stage in enumerate(('initial', 'final')):
                state = checkpoint[f'{stage}_receiver']
                independent_logits = numpy_forward(state, features)
                logits_error = float(np.max(np.abs(independent_logits - saved_logits[endpoint])))
                assert logits_error <= 1e-4
                assert np.array_equal(independent_logits >= 0, saved_logits[endpoint] >= 0)
                direct = scalar_metrics(saved_logits[endpoint], train['y'], q)
                independent = scalar_metrics(independent_logits, train['y'], q)
                observed = curve[0 if endpoint == 0 else 100]
                check_metrics(direct, observed)
                check_metrics(independent, observed)
                assert observed['parameter_hash'] == hash_parameters(state)
                scalar_endpoints[name].append(independent)
                endpoint_audits.append({'model': tag, 'branch': name, 'stage': stage,
                    'logits_error': logits_error, 'scalar_BCE_error': abs(direct['BCE'] - observed['BCE']),
                    'numpy_BCE_error': abs(independent['BCE'] - observed['BCE']), 'decisions_equal': True})
        comparison = {}
        for name in ('target', 'random1', 'random2', 'random3'):
            ref, branch = model['branches']['original']['curve'], model['branches'][name]['curve']
            cs = [ref[i]['BCE'] - branch[i]['BCE'] for i in range(101)]
            gs = [branch[i]['accuracy'] - ref[i]['accuracy'] for i in range(101)]
            record = model['result']['comparisons'][name]
            assert record['C'] == cs and record['G'] == gs
            assert (record['C0'], record['C100'], record['G100']) == (cs[0], cs[-1], gs[-1])
            assert record['first_C_positive'] == next((i for i in range(101) if cs[i] > 0), None)
            assert record['first_G_at_least_005'] == next((i for i in range(101) if gs[i] >= .05), None)
            gates = {'C0': cs[0] <= -.01, 'C100': cs[-1] >= .01, 'G100': gs[-1] >= .05}
            assert record['gates'] == gates and record['passes_all'] == all(gates.values())
            independent_c0 = scalar_endpoints['original'][0]['BCE'] - scalar_endpoints[name][0]['BCE']
            independent_c100 = scalar_endpoints['original'][1]['BCE'] - scalar_endpoints[name][1]['BCE']
            independent_g100 = scalar_endpoints[name][1]['accuracy'] - scalar_endpoints['original'][1]['accuracy']
            assert [independent_c0 <= -.01, independent_c100 >= .01, independent_g100 >= .05] == list(gates.values())
            comparison[name] = cs[-1]
        advantage = comparison['target'] - math.fsum(comparison[name] for name in ('random1', 'random2', 'random3')) / 3
        assert abs(advantage - model['result']['selection_advantage']) <= 1e-12
        assert (advantage >= .01) == model['result']['selection_advantage_passes']
        independent_advantage = math.fsum(scalar_endpoints[name][1]['BCE'] for name in ('random1', 'random2', 'random3')) / 3 - scalar_endpoints['target'][1]['BCE']
        assert (independent_advantage >= .01) == model['result']['selection_advantage_passes']
        assert model['result']['target_passes'] == model['result']['comparisons']['target']['passes_all']
        models.append({'model': tag, 'passed': True, 'primary_model_passes': model['result']['target_passes'],
                       'selection_advantage_passes': advantage >= .01, 'candidates_reconstructed': True})
    pass_count = sum(m['primary_model_passes'] for m in models)
    secondary_count = sum(m['selection_advantage_passes'] for m in models)
    assert len(models) == 6 and summary['primary_pass_count'] == pass_count
    assert summary['primary_passed'] == (pass_count >= 5)
    assert summary['selection_advantage_count'] == secondary_count
    assert summary['selection_advantage_gate_passed'] == (secondary_count >= 5)
    result = {'passed': True, 'models': models, 'endpoints': endpoint_audits,
              'scope': 'Independent algorithms by same agent; no external replication or training replay'}
    (output / 'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result

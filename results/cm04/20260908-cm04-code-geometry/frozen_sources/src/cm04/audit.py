"""Independent table, initialization, forward, scalar loss and gate audit."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def parameter_hash(state):
    h = hashlib.sha256()
    for key, value in state.items():
        h.update(key.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def forward(state, x):
    z = x.astype(np.float64)
    for i in (0, 2, 4):
        z = z @ state[f'net.{i}.weight'].numpy().astype(np.float64).T + state[f'net.{i}.bias'].numpy().astype(np.float64)
        if i != 4:
            z = np.maximum(z, 0)
    return z


def scalar_metrics(logits, labels, q):
    losses, correct = {0: [], 1: []}, {0: [], 1: []}
    for row, label, qi in zip(logits, labels, q):
        terms, ok = [], True
        for z, y in zip(row, label):
            z, y = float(z), int(y)
            terms.append(max(z, 0.) - y*z + math.log1p(math.exp(-abs(z))))
            ok = ok and ((z >= 0) == bool(y))
        losses[int(qi)].append(math.fsum(terms) / 4)
        correct[int(qi)].append(int(ok))
    return {'BCE': math.fsum(losses[0] + losses[1]) / len(q),
            'accuracy': sum(correct[0] + correct[1]) / len(q),
            'by_q': [{'q': qi, 'BCE': math.fsum(losses[qi]) / len(losses[qi]),
                      'accuracy': sum(correct[qi]) / len(correct[qi])} for qi in (0, 1)]}


def audit(output):
    output = Path(output)
    locked = json.loads((output / 'encodings.lock.json').read_text())
    maps = {'E0': list(range(16))}
    for k in (1, 2, 3):
        maps[f'E{k}'] = np.random.Generator(np.random.PCG64(84000+k)).permutation(16).tolist()
    assert maps == locked['mappings']
    records = [(q, v, r) for q in (0, 1) for v in range(16) for r in range(16)]
    qvalues = np.array([r[0] for r in records])
    labels = np.array([[((v+r) % 16 >> b) & 1 for b in range(4)] for q, v, r in records], dtype=np.float32)
    with np.load(output / 'training_table.npz', allow_pickle=False) as a:
        assert np.array_equal(a['q'], qvalues)
        assert np.array_equal(a['V'], [r[1] for r in records])
        assert np.array_equal(a['r'], [r[2] for r in records])
        assert np.array_equal(a['y'], labels)
    symbolic = json.loads((output / 'symbolic_checks.json').read_text())
    inputs = {}
    for name, mapping in maps.items():
        joint = np.zeros((2, 16, 16), dtype=int)
        for qi, v, r in records:
            m = mapping[v]
            assert (mapping.index(m) + r) % 16 == (v+r) % 16
            joint[qi, v, m] += 1
        assert np.all(joint.sum(axis=1) == 16) and np.all(joint.sum(axis=2) == 16)
        assert np.array_equal(joint, symbolic[name]['joint_counts'])
        for qi in (0, 1):
            hm = -sum((n/256)*math.log2(n/256) for n in joint[qi].sum(axis=0) if n)
            mutual = sum((n/256)*math.log2((n/256)/((1/16)*(1/16))) for n in joint[qi].ravel() if n)
            assert hm == mutual == symbolic[name]['H_M_given_q'] == symbolic[name]['I_V_M_given_q'] == 4
        for representation in ('bit', 'onehot'):
            x = np.zeros((512, 22), dtype=np.float32)
            for i, (qi, v, r) in enumerate(records):
                m = mapping[v]
                if representation == 'bit':
                    x[i, :4] = [(m >> b) & 1 for b in range(4)]
                    x[i, 8:12] = 1
                else:
                    x[i, m] = 1
                x[i, 16+qi] = 1
                x[i, 18:] = [(r >> b) & 1 for b in range(4)]
            inputs[f'{representation}_{name}'] = x
    with np.load(output / 'inputs.lock.npz', allow_pickle=False) as a:
        assert set(a.files) == set(inputs)
        for name, x in inputs.items():
            assert np.array_equal(a[name], x)
    summary = json.loads((output / 'summary.pending.json').read_text())
    count = {'E1': 0, 'E2': 0, 'E3': 0}
    seed_audits, endpoint_audits = [], []
    for seed in range(60, 90):
        torch.manual_seed(seed)
        fresh = torch.nn.Sequential(torch.nn.Linear(22, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))
        recreated = {'net.' + k: v for k, v in fresh.state_dict().items()}
        origin = torch.load(output / f'initial_{seed}.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(origin[k], v) for k, v in recreated.items())
        record = json.loads((output / f'seed_{seed}.json').read_text())
        endpoints, independent_metrics = {}, {}
        for branch, x in inputs.items():
            representation, encoding = branch.split('_')
            weights = torch.load(output / f'{seed}_{branch}_weights.pt', map_location='cpu', weights_only=True)
            start = weights['initial_receiver']
            for key in origin:
                if key == 'net.0.weight' and representation == 'onehot':
                    for v in range(16):
                        assert torch.equal(start[key][:, maps[encoding][v]], origin[key][:, v])
                    assert torch.equal(start[key][:, 16:], origin[key][:, 16:])
                else:
                    assert torch.equal(start[key], origin[key])
            assert weights['initial_optimizer']['state'] == {}
            for stage in ('initial', 'final'):
                groups = weights[f'{stage}_optimizer']['param_groups']
                assert len(groups) == 1
                group = groups[0]
                assert group['lr'] == .001 and group['betas'] == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == 0
            assert len(weights['final_optimizer']['state']) == 6
            assert all(int(s['step']) == 100 for s in weights['final_optimizer']['state'].values())
            info = json.loads((output / f'{seed}_{branch}_curve.json').read_text())
            assert info == record['branches'][branch]
            assert info['batch_rows'] == 512 and info['steps'] == 100 and info['optimizer_initial_state_count'] == 0
            curve = info['curve']
            assert [r['step'] for r in curve] == list(range(101))
            for row in curve:
                assert len(row['parameter_hash']) == 64 and math.isfinite(row['BCE']) and row['BCE'] >= 0
                assert 0 <= row['accuracy'] <= 1
                assert abs(row['BCE'] - sum(q['BCE'] for q in row['by_q']) / 2) <= 1e-6
                assert row['accuracy'] == sum(q['accuracy'] for q in row['by_q']) / 2
            with np.load(output / f'{seed}_{branch}_endpoints.npz', allow_pickle=False) as a:
                logits = a['logits']
            assert logits.shape == (2, 512, 4)
            endpoints[branch] = logits
            independent_metrics[branch] = []
            for endpoint, stage in enumerate(('initial', 'final')):
                state = weights[f'{stage}_receiver']
                independent_logits = forward(state, x)
                error = float(np.abs(independent_logits - logits[endpoint]).max())
                assert error <= 1e-4
                assert np.array_equal(independent_logits >= 0, logits[endpoint] >= 0)
                direct = scalar_metrics(logits[endpoint], labels, qvalues)
                independent = scalar_metrics(independent_logits, labels, qvalues)
                observed = curve[0 if endpoint == 0 else 100]
                for value in (direct, independent):
                    assert abs(value['BCE'] - observed['BCE']) <= 1e-6
                    assert value['accuracy'] == observed['accuracy']
                    for qi in (0, 1):
                        assert abs(value['by_q'][qi]['BCE'] - observed['by_q'][qi]['BCE']) <= 1e-6
                        assert value['by_q'][qi]['accuracy'] == observed['by_q'][qi]['accuracy']
                assert parameter_hash(state) == observed['parameter_hash']
                endpoint_audits.append({'seed': seed, 'branch': branch, 'stage': stage, 'logits_error': error,
                    'scalar_BCE_error': abs(direct['BCE']-observed['BCE']),
                    'numpy_BCE_error': abs(independent['BCE']-observed['BCE']), 'decisions_equal': True})
                independent_metrics[branch].append(independent)
        controls_ok = True
        for name in ('E1', 'E2', 'E3'):
            base, arm = record['branches']['bit_E0']['curve'], record['branches']['bit_'+name]['curve']
            loss = [b['BCE']-a['BCE'] for a, b in zip(base, arm)]
            acc = [a['accuracy']-b['accuracy'] for a, b in zip(base, arm)]
            result = record['result']['bit_comparisons'][name]
            assert result['D_loss'] == loss and result['D_acc'] == acc
            assert result['D_loss100'] == loss[-1] and result['D_acc100'] == acc[-1]
            gates = (loss[-1] >= .01, acc[-1] >= .05)
            assert (result['loss_gate'], result['accuracy_gate']) == gates
            assert result['joint_pass'] == all(gates)
            independent_loss = independent_metrics['bit_'+name][1]['BCE'] - independent_metrics['bit_E0'][1]['BCE']
            independent_acc = independent_metrics['bit_E0'][1]['accuracy'] - independent_metrics['bit_'+name][1]['accuracy']
            assert gates == (independent_loss >= .01, independent_acc >= .05), 'Numerically uncertain bit gate'
            count[name] += all(gates)
            reference, other = record['branches']['onehot_E0']['curve'], record['branches']['onehot_'+name]['curve']
            dl = [abs(a['BCE']-b['BCE']) for a, b in zip(reference, other)]
            da = [abs(a['accuracy']-b['accuracy']) for a, b in zip(reference, other)]
            dz = [float(np.max(np.abs(a-b))) for a, b in zip(endpoints['onehot_E0'], endpoints['onehot_'+name])]
            violations = [{'step': i, 'BCE_diff': dl[i], 'accuracy_diff': da[i]} for i in range(101) if dl[i] > 1e-5 or da[i] > 1e-5]
            expected_control = {'max_BCE_diff': max(dl), 'max_accuracy_diff': max(da), 'initial_logits_diff': dz[0],
                'final_logits_diff': dz[1], 'trajectory_violations': violations,
                'passed': not violations and max(dz) <= 1e-4}
            assert expected_control == record['result']['onehot_controls'][name]
            controls_ok = controls_ok and expected_control['passed']
        assert record['result']['control_passed'] == controls_ok
        seed_audits.append({'seed': seed, 'audit_passed': True, 'control_passed': controls_ok})
    controls = all(row['control_passed'] for row in seed_audits)
    assert summary['bit_joint_pass_counts'] == count
    assert summary['bit_numeric_gate_passed'] == all(n >= 24 for n in count.values())
    assert summary['control_passed'] == controls and summary['mechanism_claim_allowed'] == controls
    assert summary['primary_passed'] == (controls and all(n >= 24 for n in count.values()))
    result = {'passed': True, 'seeds': seed_audits, 'endpoints': endpoint_audits,
              'scope': 'Independent algorithms by same agent; no external replication or training replay'}
    (output / 'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result

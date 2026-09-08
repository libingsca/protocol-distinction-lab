import hashlib

import numpy as np
import torch


ENCODINGS = ('E0', 'E1', 'E2', 'E3')
REPRESENTATIONS = ('bit', 'onehot')
ADAM = {'lr': .001, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': 0}


class Receiver(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(22, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))

    def forward(self, inputs):
        return self.net(inputs)


def state_hash(state):
    h = hashlib.sha256()
    for key, value in state.items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def table():
    q, v, r = np.meshgrid(np.arange(2), np.arange(16), np.arange(16), indexing='ij')
    q, v, r = q.ravel(), v.ravel(), r.ravel()
    y = (((v + r) % 16)[:, None] >> np.arange(4)) & 1
    return {'q': q, 'V': v, 'r': r, 'y': y.astype(np.float32)}


def encodings():
    return {'E0': list(range(16)), **{f'E{k}': np.random.Generator(np.random.PCG64(84000+k)).permutation(16).tolist() for k in (1, 2, 3)}}


def features(data, mapping, representation):
    m = np.asarray(mapping)[data['V']]
    suffix = np.column_stack([data['q'] == 0, data['q'] == 1, (data['r'][:, None] >> np.arange(4)) & 1])
    if representation == 'onehot':
        message = np.eye(16)[m]
    elif representation == 'bit':
        message = np.column_stack([(m[:, None] >> np.arange(4)) & 1,
            np.zeros((len(m), 4)), np.tile([1, 1, 1, 1, 0, 0, 0, 0], (len(m), 1))])
    else:
        raise ValueError(representation)
    return np.column_stack([message, suffix]).astype(np.float32)


def paired_state(origin, mapping, representation):
    state = {key: value.clone() for key, value in origin.items()}
    if representation == 'onehot':
        assert sorted(mapping) == list(range(16))
        state['net.0.weight'][:, mapping] = origin['net.0.weight'][:, :16]
    else:
        assert representation == 'bit'
    return state


def symbolic_check(data, maps):
    checks = {}
    for name, mapping in maps.items():
        assert sorted(mapping) == list(range(16))
        m = np.asarray(mapping)[data['V']]
        inverse = np.argsort(mapping)
        answers = (inverse[m] + data['r']) % 16
        assert np.array_equal((answers[:, None] >> np.arange(4)) & 1, data['y'])
        joint = np.zeros((2, 16, 16), dtype=int)
        np.add.at(joint, (data['q'], data['V'], m), 1)
        assert np.all(joint.sum(axis=1) == 16) and np.all(joint.sum(axis=2) == 16)
        assert np.all((joint > 0).sum(axis=2) == 1)
        checks[name] = {'symbolic_accuracy': 1., 'joint_counts': joint.tolist(),
                        'H_M_given_q': 4., 'I_V_M_given_q': 4.}
    return checks


def train(initial, inputs, q, y, steps, destination=None):
    model = Receiver()
    model.load_state_dict(initial)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), **ADAM)
    assert not optimizer.state
    empty_optimizer = optimizer.state_dict()
    initial_hash = state_hash(initial)
    curve, endpoints = [], []
    for step in range(steps + 1):
        logits = model(inputs)
        row_losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none').mean(dim=1)
        loss = row_losses.mean()
        correct = ((logits.detach() >= 0) == y.bool()).all(dim=1).float()
        assert torch.isfinite(loss)
        curve.append({'step': step, 'BCE': float(loss.detach()), 'accuracy': float(correct.mean()),
            'by_q': [{'q': qi, 'BCE': float(row_losses[q == qi].mean().detach()),
                      'accuracy': float(correct[q == qi].mean())} for qi in (0, 1)],
            'parameter_hash': state_hash(model.state_dict())})
        if step in (0, steps):
            endpoints.append(logits.detach().numpy().copy())
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
    assert all(int(s['step']) == steps for s in optimizer.state.values())
    assert state_hash(initial) == initial_hash
    if destination:
        torch.save({'initial_receiver': initial, 'final_receiver': model.state_dict(),
                    'initial_optimizer': empty_optimizer, 'final_optimizer': optimizer.state_dict()}, destination)
    return {'batch_rows': len(y), 'steps': steps, 'initial_hash': initial_hash,
            'optimizer_initial_state_count': 0, 'curve': curve}, np.array(endpoints)


def compare_seed(branches, endpoints):
    bit, controls = {}, {}
    for name in ENCODINGS[1:]:
        baseline, other = branches['bit_E0']['curve'], branches[f'bit_{name}']['curve']
        dloss = [b['BCE'] - a['BCE'] for a, b in zip(baseline, other)]
        dacc = [a['accuracy'] - b['accuracy'] for a, b in zip(baseline, other)]
        bit[name] = {'D_loss': dloss, 'D_acc': dacc, 'D_loss100': dloss[-1], 'D_acc100': dacc[-1],
                     'loss_gate': dloss[-1] >= .01, 'accuracy_gate': dacc[-1] >= .05,
                     'joint_pass': dloss[-1] >= .01 and dacc[-1] >= .05}
        ref, arm = branches['onehot_E0']['curve'], branches[f'onehot_{name}']['curve']
        loss_diffs = [abs(a['BCE'] - b['BCE']) for a, b in zip(ref, arm)]
        accuracy_diffs = [abs(a['accuracy'] - b['accuracy']) for a, b in zip(ref, arm)]
        logits_diffs = np.abs(endpoints['onehot_E0'] - endpoints[f'onehot_{name}']).max(axis=(1, 2))
        violations = [{'step': i, 'BCE_diff': loss_diffs[i], 'accuracy_diff': accuracy_diffs[i]}
                      for i in range(len(ref)) if loss_diffs[i] > 1e-5 or accuracy_diffs[i] > 1e-5]
        controls[name] = {'max_BCE_diff': max(loss_diffs), 'max_accuracy_diff': max(accuracy_diffs),
            'initial_logits_diff': float(logits_diffs[0]), 'final_logits_diff': float(logits_diffs[1]),
            'trajectory_violations': violations,
            'passed': not violations and bool(np.all(logits_diffs <= 1e-4))}
    return {'bit_comparisons': bit, 'onehot_controls': controls,
            'control_passed': all(c['passed'] for c in controls.values())}


def summarize(reports):
    assert len(reports) == 30 and [r['seed'] for r in reports] == list(range(60, 90))
    counts = {name: sum(r['result']['bit_comparisons'][name]['joint_pass'] for r in reports) for name in ENCODINGS[1:]}
    controls = all(r['result']['control_passed'] for r in reports)
    numeric_gate = all(n >= 24 for n in counts.values())
    return {'bit_joint_pass_counts': counts, 'bit_numeric_gate_passed': numeric_gate,
            'control_passed': controls, 'mechanism_claim_allowed': controls,
            'primary_passed': controls and numeric_gate,
            'branches_completed': 240, 'updates_completed': 24000}

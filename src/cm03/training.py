import hashlib

import numpy as np
import torch

from .protocols import BRANCHES


ADAM = {'lr': .001, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': 0}


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in state.items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def message_tensor(ids):
    bits = ((np.asarray(ids)[:, None] >> np.arange(4)) & 1).astype(np.float32)
    return torch.from_numpy(np.concatenate([bits, np.zeros_like(bits)], axis=1))


def metrics(logits, labels, q):
    with torch.no_grad():
        row_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none').mean(dim=1)
        row_correct = ((logits >= 0) == labels.bool()).all(dim=1).float()
        return {'BCE': float(row_loss.mean()), 'accuracy': float(row_correct.mean()),
                'by_q': [{'q': qi, 'BCE': float(row_loss[q == qi].mean()),
                          'accuracy': float(row_correct[q == qi].mean())} for qi in (0, 1)]}


def train_branch(receiver, initial, message, q, r, y, steps, output=None):
    receiver.load_state_dict(initial)
    receiver.train()
    for p in receiver.parameters():
        p.requires_grad_(True)
    initial_hash = state_hash(initial)
    assert state_hash(receiver.state_dict()) == initial_hash
    optimizer = torch.optim.Adam(receiver.parameters(), **ADAM)
    assert len(optimizer.state) == 0
    optimizer_start = optimizer.state_dict()
    curve, endpoints = [], []
    for step in range(steps + 1):
        logits = receiver(message, q, r)
        row = {'step': step, **metrics(logits, y, q), 'parameter_hash': state_hash(receiver.state_dict())}
        curve.append(row)
        if step in (0, steps):
            endpoints.append(logits.detach().cpu().numpy().copy())
        if step < steps:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in receiver.parameters())
            optimizer.step()
    assert all(int(value['step']) == steps for value in optimizer.state.values())
    if output is not None:
        torch.save({'initial_receiver': initial, 'final_receiver': receiver.state_dict(),
                    'initial_optimizer': optimizer_start, 'final_optimizer': optimizer.state_dict()}, output)
    return {'steps': steps, 'batch_rows': len(y), 'initial_hash': initial_hash,
            'optimizer_initial_state_count': 0, 'curve': curve}, np.array(endpoints)


def compare(branches):
    reference = branches['original']['curve']
    compared = {}
    for name in BRANCHES[1:]:
        branch = branches[name]['curve']
        assert len(reference) == len(branch) == 101
        credit = [a['BCE'] - b['BCE'] for a, b in zip(reference, branch)]
        gain = [b['accuracy'] - a['accuracy'] for a, b in zip(reference, branch)]
        gates = {'C0': credit[0] <= -.01, 'C100': credit[100] >= .01, 'G100': gain[100] >= .05}
        compared[name] = {'C': credit, 'G': gain, 'C0': credit[0], 'C100': credit[100], 'G100': gain[100],
                          'first_C_positive': next((i for i, x in enumerate(credit) if x > 0), None),
                          'first_G_at_least_005': next((i for i, x in enumerate(gain) if x >= .05), None),
                          'gates': gates, 'passes_all': all(gates.values())}
    advantage = compared['target']['C100'] - float(np.mean([compared[name]['C100'] for name in BRANCHES[2:]]))
    return {'comparisons': compared, 'target_passes': compared['target']['passes_all'],
            'selection_advantage': advantage, 'selection_advantage_passes': advantage >= .01}

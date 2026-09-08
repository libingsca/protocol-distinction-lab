import numpy as np


BRANCHES = ('original', 'target', 'random1', 'random2', 'random3')


def validate_training(train):
    assert set(train) == {'x', 'q', 'r', 'y'}
    assert train['x'].shape == (4096, 8) and train['q'].shape == (4096,)
    assert train['r'].shape == train['y'].shape == (4096, 4)
    assert all(np.isin(value, [0, 1]).all() for value in train.values())
    q = train['q'].astype(int)
    r = (train['r'] * 2**np.arange(4)).sum(axis=1).astype(int)
    v = ((train['y'] * 2**np.arange(4)).sum(axis=1).astype(int) - r) % 16
    _, first, inverse, counts = np.unique(np.column_stack([train['x'], q]), axis=0,
                                           return_index=True, return_inverse=True, return_counts=True)
    assert len(first) == 256 and np.all(counts == 16)
    for g in range(len(first)):
        rows = inverse == g
        assert sorted(r[rows]) == list(range(16)) and len(set(v[rows])) == 1
    for qi in range(2):
        assert set(v[q == qi]) == set(range(16))
    return v, first


def make_random(original, target, seed, max_attempts=10000):
    original, target = np.asarray(original), np.asarray(target)
    assert sorted(target.tolist()) == list(range(len(target)))
    changed = np.flatnonzero(original != target)
    rng = np.random.Generator(np.random.PCG64(seed))
    for attempt in range(1, max_attempts + 1):
        proposal = target.copy()
        proposal[changed] = rng.permutation(target[changed])
        if np.all(proposal[changed] != original[changed]) and not np.array_equal(proposal, target):
            return proposal, {'seed': seed, 'attempts': attempt, 'changed_V': changed.tolist()}
    raise ValueError(f'No eligible random permutation within {max_attempts} attempts; seed={seed}')


def freeze_model(index, q, v, first, original_rows, target_map):
    original_map = np.empty((2, 16), dtype=int)
    for qi in range(2):
        for vi in range(16):
            options = np.unique(original_rows[(q == qi) & (v == vi)])
            assert len(options) == 1
            original_map[qi, vi] = options[0]
    target_map = np.asarray(target_map, dtype=int)
    assert target_map.shape == (2, 16)
    maps = {'original': original_map, 'target': target_map}
    generation = {}
    for k in range(3):
        values, records = [], []
        for qi in range(2):
            candidate, record = make_random(original_map[qi], target_map[qi], 73001 + 100*index + 10*qi + k)
            values.append(candidate)
            records.append({'q': qi, **record})
        maps[f'random{k+1}'] = np.array(values)
        generation[f'random{k+1}'] = records
    rows = {name: mapping[q, v] for name, mapping in maps.items()}
    assert np.array_equal(rows['original'], original_rows)
    distances = {}
    for name in BRANCHES:
        xor = rows[name] ^ original_rows
        bit_distance = ((xor[:, None] >> np.arange(4)) & 1).sum(axis=1)
        distances[name] = []
        for qi in range(2):
            selected = q == qi
            group_rows = first[q[first] == qi]
            distances[name].append({'q': qi,
                'changed_group_fraction': float(np.mean(rows[name][group_rows] != original_rows[group_rows])),
                'mean_bit_hamming': float(bit_distance[selected].mean()),
                'changed_V': np.flatnonzero(maps[name][qi] != original_map[qi]).tolist()})
        if name != 'original':
            for qi in range(2):
                assert sorted(maps[name][qi]) == list(range(16))
            assert np.array_equal(rows[name] != original_rows, rows['target'] != original_rows)
    duplicates = [[a, b] for ai, a in enumerate(BRANCHES[2:]) for b in BRANCHES[2:][ai+1:]
                  if np.array_equal(maps[a], maps[b])]
    return {'maps': {name: value.tolist() for name, value in maps.items()},
            'random_generation': generation, 'distances': distances,
            'duplicate_random_branches': duplicates}, rows

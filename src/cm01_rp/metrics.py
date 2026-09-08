import numpy as np


def bits():
    return ((np.arange(16)[:, None] >> np.arange(4)) & 1)


def distributions(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape[-1] != 4 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected four finite Bernoulli probabilities')
    return np.prod(np.where(bits().astype(bool), p[..., None, :], 1 - p[..., None, :]), axis=-1)


def distances(probabilities):
    joint = distributions(probabilities)  # q,r,m,answer
    a, b = joint[:, :, :, None, :], joint[:, :, None, :, :]
    tv = .5 * np.abs(a - b).sum(axis=-1)
    midpoint = (a + b) / 2
    def kl(x):
        ratio = np.divide(x, midpoint, out=np.ones_like(midpoint), where=(x > 0) & (midpoint > 0))
        return np.sum(x * np.log2(ratio), axis=-1)
    js = .5 * (kl(a) + kl(b))
    return joint, tv, js


def diagnose(logits, usage, required, mean_limit=.05, max_limit=.10):
    logits = np.asarray(logits)
    if logits.shape != (2, 16, 16, 4) or not np.isfinite(logits).all():
        raise ValueError('Expected q,r,message,bit logits')
    probabilities = 1 / (1 + np.exp(-logits.astype(np.float64)))
    joint, tv, js = distances(probabilities)
    predicted = ((logits >= 0) * (2 ** np.arange(4))).sum(axis=-1)
    decoded = (predicted - np.arange(16)[None, :, None]) % 16
    contexts = []
    for q in range(2):
        coverage = (decoded[q, :, :, None] == np.arange(16)).sum(axis=0)
        mean, maximum = tv[q].mean(axis=0), tv[q].max(axis=0)
        pairs = [[i, j] for i in range(16) for j in range(i + 1, 16)
                 if mean[i, j] <= mean_limit and maximum[i, j] <= max_limit]
        occupied = [[i, j] for i, j in pairs if usage[q, i] > 0 and usage[q, j] > 0]
        missing = [v for v in required[q] if coverage[:, v].max() < 16]
        closest = min(((float(mean[i,j]),i,j) for i in range(16) for j in range(i+1,16)))
        contexts.append({'q': q, 'usage': usage[q].tolist(), 'necessary_v': list(map(int, required[q])),
            'near_pairs': pairs, 'occupied_near_pairs': occupied, 'missing_v': list(map(int, missing)),
            'best_correct_r_by_v': coverage.max(axis=0).tolist(), 'coverage_counts': coverage.tolist(),
            'closest_pair': {'messages': list(closest[1:]), 'mean_tv': closest[0],
                             'max_tv': float(maximum[closest[1],closest[2]])},
            'positive': bool(occupied and missing)})
    return {'contexts': contexts, 'positive': any(c['positive'] for c in contexts)}, {
        'probabilities': probabilities, 'joint_probabilities': joint, 'tv': tv, 'js_bits': js, 'decoded_v': decoded}

import numpy as np


def create_greedy_mediaplan_for_parent_pool(l, n, seed):
    np.random.seed(seed)
    sum_max_reach = sum(max(levels, key=lambda x: x.imps).imps for desc, levels in l)
    out = []
    for _ in range(n):
        rnd = np.random.random()
        r = sum_max_reach * rnd
        # print(f"initial r: {r: 4.2f}")
        np.random.shuffle(l)
        choices = []
        for desc, levels in l:
            blah_ = [x for x in sorted([x.asDict() for x in levels], key=lambda x: x['imps']) if x['imps'] < r]
            if len(blah_) == 0:
                continue

            choice = blah_[-1]
            choice['descendent'] = desc
            choices.append(choice)
            r = r - choices[-1]['imps']
        out.append((str(rnd)[2:], choices))
    return out


def create_random_mediaplan_for_parent_pool(l, n, seed):
    np.random.seed(seed)
    sum_max_reach = sum(max(levels, key=lambda x: x.imps).imps for desc, levels in l)
    out = []
    for _ in range(n):
        rnd = np.random.random()
        r = sum_max_reach * rnd
        # print(f"initial r: {r: 4.2f}")
        np.random.shuffle(l)
        choices = []
        for desc, levels in l:
            blah_ = [x for x in sorted([x.asDict() for x in levels], key=lambda x: x['imps']) if x['imps'] < r]
            if len(blah_) == 0:
                continue

            choice = np.random.choice(blah_)
            choice['descendent'] = desc
            choices.append(choice)
            r = r - choices[-1]['imps']
        out.append((str(rnd)[2:], choices))
    return out


def create_jitter_uniform_mediaplan_for_parent_pool(l, n, seed):
    np.random.seed(seed)
    sum_max_reach = sum(max(levels, key=lambda x: x.imps).imps for desc, levels in l)
    out = []
    for _ in range(n):
        # rnd = np.random.rand()
        rnd = np.random.beta(1, 4)
        stddev = np.random.exponential(.05)
        np.random.shuffle(l)
        choices = []
        for i, (desc, levels) in enumerate(l):
            S = sorted([x.asDict() for x in levels], key=lambda x: x['imps'])
            r = np.random.normal(rnd, stddev)
            r = np.clip(r, 1e-5, 1 - 1e-5)
            if i == 0:
                n = int(len(S) * r)
            else:
                n = int((len(S) + 1) * r)
                if n == 0:
                    continue
                n -= 1
            choice = dict(S[n])
            choice['descendent'] = desc
            choices.append(choice)
        out.append((str(rnd)[2:], choices))
    return out


def create_all_mediaplan_for_parent_pool(l, n, seed):
    np.random.seed(seed)
    p = np.random.rand()
    if p < .1:
        return create_greedy_mediaplan_for_parent_pool(l, n, seed)
    elif p < .2:
        return create_random_mediaplan_for_parent_pool(l, n, seed)
    else:
        return create_jitter_uniform_mediaplan_for_parent_pool(l, n, seed)
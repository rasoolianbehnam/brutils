import itertools
import operator
import sys
from functools import reduce

import pandas as pd

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, '..')
from brutils.misc.xr_factor import XrFactor
Factor = XrFactor


def adj_matrix_to_adj_list(matrix):
    edges = {}
    for i in range(len(matrix)):
        nbs = set()
        for j in range(len(matrix)):
            if matrix[i, j] == 1:
                nbs.add(j)
        edges[i] = nbs
    return edges


def eliminate_var(F, adj_list, next_var, skeleton):
    use_factors, non_use_factors = [], []
    scope = set()
    for i, f in enumerate(F):
        if next_var in f.vars:
            use_factors.append(i)
            scope.update(f.vars)
        else:
            non_use_factors.append(i)
    scope = sorted(scope)

    for i, j in itertools.permutations(scope, 2):
        if i not in adj_list:
            adj_list[i] = {j, }
        else:
            adj_list[i].add(j)

    # next steps removes the next_var from adj_list
    for k in adj_list:
        if next_var in adj_list[k]:
            adj_list[k].remove(next_var)
    del adj_list[next_var]

    newF, newmap = [], {}
    for i in non_use_factors:
        newmap[i] = len(newF)
        newF.append(F[i])

    new_factor = Factor(variables=[], value=[])
    for i in use_factors:
        new_factor = new_factor * F[i]  # Since this just a simulation, we don't really need to compute values. So @
    new_factor = new_factor.marginalise({next_var, })
    newF.append(new_factor)

    for i in range(len(skeleton['nodes'])):
        if skeleton['factor_idxs'][i] in use_factors:
            skeleton['edges'].append((skeleton['nodes'][i], set(scope)))
            skeleton['factor_idxs'][i] = None
        elif skeleton['factor_idxs'][i] is not None:
            skeleton['factor_idxs'][i] = newmap[skeleton['factor_idxs'][i]]
    skeleton['nodes'].append(set(scope))
    skeleton['factor_idxs'].append(len(newF) - 1)

    return newF


def prune_tree(skeleton):
    found = True
    while found:
        found = False

        for u, v in skeleton['edges']:
            if u.issuperset(v):
                found = True
                parent = u
                child = v
                break
            elif v.issuperset(u):
                found = True
                parent = v
                child = u
                break

        if not found:
            break

        new_edges = []
        for u, v in skeleton['edges']:
            if (u, v) == (child, parent) or (v, u) == (child, parent):
                continue
            elif u == child:
                new_edges.append((parent, v))
            elif v == child:
                new_edges.append((u, parent))
            else:
                new_edges.append((u, v))
        skeleton['edges'] = new_edges
        skeleton['nodes'] = [node for node in skeleton['nodes'] if node != child]


def create_clique_tree(factors, evidence=None):
    V, domains = set(), dict()
    for factor in factors:
        V.update(factor.vars)
        for v, d in zip(factor.vars, factor.domains):
            if v in domains:
                assert np.all(domains[v] == d), "Domain mismatch between factors"
            else:
                domains[v] = d

    adj_list = {v: {v, } for v in V}
    for factor in factors:
        for u, v in itertools.permutations(factor.vars, 2):
            adj_list[u].add(v)

    cliques_considered = 0
    F = factors
    skeleton = {'nodes': [], 'factor_idxs': [], 'edges': [], 'factor_list': factors}
    while cliques_considered < len(V):
        next_var = min(adj_list, key=lambda x: len(adj_list[x]))
        F = eliminate_var(F, adj_list, next_var, skeleton)
        cliques_considered += 1
        if not adj_list:
            break

    prune_tree(skeleton)
    tree = compute_initial_potentials(skeleton)
    if evidence:
        for i, f in enumerate(tree['clique_list']):
            tree['clique_list'][i] = f.evidence(evidence)

    return tree


def compute_initial_potentials(skeleton):
    """
    Args:
        skeleton: a dictionary with following keys.
            'nodes': a list of sets. Each set is a set of constituent variables. e.g. {1,2,3}
            'edges': a list of edges. A single element would look like ({1,2,3}, {2,3,4})
                which means there is an edge between node {1,2,3} and node {2,3,4}. If (a, b) is
                in the list, (b, a) will not be in the list.
            'factor_list': a list of initialized Factors.

    Returns:
        a dict with ['clique_list', 'edges'] keys.
        'clique_list': a list of factors associated with each clique
        'adj_list': adjacency list with integer nodes. adj_list[0] = {1,2}
            implies that there is an edges  clique_list[0]-clique_list[1]
            and clique_list[0]-clique_list[2]
    """
    # n = len(skeleton['nodes'])
    # var_domain = {}
    # for factor in skeleton['factor_list']:
    #     for var, domain in zip(factor.vars, factor.domains):
    #         var_domain[var] = domain

    # clique_list = []
    # for clique in skeleton['nodes']:
    #     clique = sorted(clique)
    #     domains = [var_domain[v] for v in clique]
    #     clique_list.append(Factor(clique, domains, init=1))
    # adj_list = {i: set() for i in range(n)}

    # Solution Start
    adj_list = _get_adj_list(skeleton)
    clique_list = _get_clique_list(skeleton)

    # Solution End

    return {'clique_list': clique_list, 'adj_list': adj_list}


def _get_clique_list(skeleton):
    factor_list = skeleton['factor_list']
    nodes = skeleton['nodes']
    factor_vars = [set(x.vars) for x in factor_list]
    cost = np.ones((len(nodes), len(factor_vars)))
    for i, node in enumerate(nodes):
        for j, var in enumerate(factor_vars):
            if set(var).issubset(node):
                cost[i, j] = 0
    row_ind, col_ind = linear_sum_assignment(cost)
    remaining_vars = set(range(len(factor_vars))) - set(col_ind)
    ff = [{j} for j in col_ind]
    for j in remaining_vars:
        for i, node in enumerate(nodes):
            if factor_vars[j].issubset(node):
                ff[i].add(j)
                break

    def create_clique(vrs):
        factors = [factor_list[i] for i in vrs]
        return reduce(operator.mul, factors)

    clique_list = list(map(create_clique, ff))
    return clique_list


def _get_adj_list(skeleton):
    mp = {tuple(x): i for i, x in enumerate(skeleton['nodes'])}
    a = pd.DataFrame(skeleton['edges']).applymap(tuple).applymap(mp.get)
    b = a.copy()
    b.columns = [1, 0]
    adj_list = pd.concat([a, b]).groupby(0).agg(lambda s: set(s))[1].to_dict()
    return adj_list


def get_next_clique(clique_tree, msgs):
    """
    Args:
        clique_tree: a structure returned by `compute_initial_potentials`
        msgs: A dictionary of dictionary.
            If u has sent message to v, that msg will be msgs[v][u].

    Returns:
        a tuple (i, j) if i is ready to send the message to j.
        If all the messages has been passed, return None.

        If more than one message is ready to be transmitted, return
        the pair (i,j) that is numerically smallest. If you use an outer
        for loop over i and an inner for loop over j, breaking when you find a
        ready pair of cliques, you will get the right answer.
    """
    adj = clique_tree['adj_list']

    # Solution Start
    n = len(adj)
    sent = {(i, j) for j in msgs for i in msgs[j]}
    for i in adj:
        for j in adj:
            if (j, i) in sent:
                continue
            if i in adj[j] and (adj[j] - set(msgs[j])).issubset({i}):
                return j, i
    # Solution End


def clique_tree_calibrate(clique_tree, is_max=0):
    # msgs[u] = {v: msg_from_v_to_u}
    adj = clique_tree['adj_list']

    # Solution Start

    # Following is a dummy line to make the grader happy when this is unimplemented.
    # Delete it or create new list `calibrated_potentials`
    n = len(clique_tree['clique_list'])
    calibrated_potentials = [None for _ in range(n)]
    if is_max == 0:
        msgs = get_all_messages(clique_tree, is_max)
        for k in msgs:
            calibrated_potentials[k] = clique_tree['clique_list'][k] * reduce(operator.mul, msgs[k].values())
    elif is_max == 1:
        log_clique_tree = {
            'clique_list': [f.log_transform() for f in clique_tree['clique_list']],
            'adj_list': clique_tree['adj_list']
        }
        msgs = get_all_messages(log_clique_tree, is_max)
        for k in msgs:
            calibrated_potentials[k] = log_clique_tree['clique_list'][k] + reduce(operator.add, msgs[k].values())
    else:
        raise ValueError("is_max should be in {0, 1}")

    # Solution End

    return {'clique_list': calibrated_potentials, 'adj_list': adj}


def get_all_messages(clique_tree, is_max):
    f = sp_message if is_max == 0 else sp_message_max
    msgs = {i: {} for i in range(len(clique_tree['clique_list']))}
    while True:
        idx = get_next_clique(clique_tree, msgs)
        if idx is None:
            break
        i, j = idx
        msgs[j][i] = f(clique_tree['clique_list'], msgs, i, j)
    return msgs


def sp_message(cliques, msgs, i, j):
    incoming = [v for k, v in msgs[i].items() if k != j]
    out, b = cliques[i], cliques[j]
    if len(incoming):
        delta = reduce(operator.mul, incoming)
        out = out * delta
    sij = set(out.vars) & set(b.vars)
    return out.marginalize_in(sij).normalize()


def sp_message_max(cliques, msgs, i, j):
    incoming = [v for k, v in msgs[i].items() if k != j]
    out, b = cliques[i], cliques[j]
    if len(incoming):
        delta = reduce(operator.add, incoming)
        out = out + delta
    sij = list(set(out.vars) & set(b.vars))
    # return out.marginalize_in(sij, op=max)
    return factor_max_marginalization_in(out, sij)


def compute_exact_marginals_bp(factors, evidence=None, is_max=0):
    """
    this function takes a list of factors, evidence, and a flag is_max,
    runs exact inference and returns the final marginals for the
    variables in the network. If is_max is 1, then it runs exact MAP inference,
    otherwise exact inference (sum-prod).
    It returns a list of size equal to the number of variables in the
    network where M[i] represents the factor for ith variable.

    Args:
        factors: list[Factor]
        evidence: dict[variable] -> observation
        is_max: use max product algorithm

    Returns:
        list of factors. Each factor should have only one variable.
    """

    marginals = []
    if evidence is None:
        evidence = {}

    # Solution Start
    clique_tree = create_clique_tree(factors, evidence)
    calibrated_clique_tree = clique_tree_calibrate(clique_tree, is_max=is_max)
    marginals = []
    variables = set(reduce(operator.add, [x.vars for x in factors]))
    for i in variables:
        for clique in calibrated_clique_tree['clique_list']:
            if i in clique.vars:
                if is_max == 0:
                    s = clique.marginalize_in([i]).normalize()
                else:
                    s = factor_max_marginalization_in(clique, [i])
                marginals.append(s)
                break
    # Solution End
    return marginals


def factor_max_marginalization(factor, variables=None):
    if not variables or not factor.vars:
        return factor

    new_vars = sorted(set(factor.vars) - set(variables))
    if not new_vars:
        raise ValueError("Resultant factor has empty scope.")

    # Solution Start
    new_factor = factor_max_marginalization_in(factor, new_vars)
    # Solution End

    return new_factor


def factor_max_marginalization_in(factor, variables):
    return factor.max_marginalise_in(variables)


def max_decoding(marginals):
    """
    Finds the best assignment for each variable from the marginals passed in.
    Returns A such that A[i] returns the index of the best instantiation for variable i.

    For instance: Let's say we have two variables 0 and 1.
    Marginals for 0 = [0.1, 0.3, 0.6]
    Marginals for 1 = [0.92, 0.08]
    max_decoding(marginals) == [2, 0]

    M is a list of factors, where each factor is only over one variable.
    """

    A = np.zeros(len(marginals), dtype=np.int32)

    # Solution Start
    A = np.array([x.df.value.argmax() for x in marginals])
    # Solution End
    return A

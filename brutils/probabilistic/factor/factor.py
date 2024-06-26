import itertools
from functools import partial

import numpy as np
import pandas as pd


class Factor:
    """Generic class for discrete factors.
        To create a factor of two variables, x (having two possible values) and y (having 3 possible values) use,
        f = Factor(['x', 'y'], [2, 3])

        To assign or access value of assignment [x=1, y=0], you can either use
            f[1, 0] = .5 (here order of the keys must match `variables` argument in the constructor) or,
            f[{'y': 1, 'x': 0}] = .5

        Use `Factor.Z` to get nomralizing constant.

        This class supports following operators.

        Equality:
            f1 == f2 returns true if both factors have same set of variables, and
                same domain for each of the variables, and
                same value(up to six decimal places) for each assignment.

        Factor multiplication:
            ```f1 = Factor(['a', 'b', 'c'], [2, 2, 2])
               f2 = Factor(['b', 'c', 'd'], [2, 2, 3])
               initialize(f1); initialize(f2);
               f3 = f1*f2``` returns a `Factor(['a', 'b', 'c', 'd'], [2, 2, 2, 3])`

        Factor addition:
            similar to factor multiplication, except factors are added to get final factor

        Uninitialized factor multiplication:
            `f3 = f1 @ f2`, returns uninitialized factor product
    """

    def __init__(self, variables, domains, init=None, default=None):
        """
        One needs to assign factor values after initialization as constructor does not initialize assignments.

        Args:
            variables: a list of hashable names. e.g. list[str] or list[int]
            domains: list of list of elements.
                e.g. if variables[0] is "Coin_Flip", domains[0] could be ["Heads", "Tails"]
                if domains[i] is an integer, it will be converted to [0, 1, ..., domains[i]-1]
            init: float: initialize value of each assignment to init value
            default: float: if not None, when assignment is not assigned (e.g. no init) return this value.
                be careful of using this argument as not all methods can be applied on Factor object if
                this argument is not None.
        """
        if len(set(variables)) != len(variables):
            raise ValueError("Duplicate variable names are not permitted")

        if len(variables) != len(domains):
            raise ValueError("domains size must match variables list")

        self.domains = [None] * len(domains)
        for i, d in enumerate(domains):
            if isinstance(d, (int, np.integer)):
                self.domains[i] = range(d)
            else:
                self.domains[i] = domains[i]

        self.vars = variables
        self.val = {}

        if init is not None:
            for assignment in itertools.product(*self.domains):
                self.val[assignment] = init

        self.default = default

    @property
    def Z(self):
        if not self.val:
            raise ValueError("Z is undefined for an empty/uninitialized factor")
        return sum(self.val.values())

    def __repr__(self):
        return repr(self.df2)

    @property
    def df2(self):
        df = self.df
        df = df.set_index(self.vars)
        return df

    @property
    def df(self):
        tmp = []
        for k, v in self.val.items():
            tmp.append((*k, v))
        df = pd.DataFrame(tmp, columns=self.vars + ['value'])
        return df

    def __eq__(self, other):
        if set(self.vars) != set(other.vars):
            return False

        other_map = [other.vars.index(var) for var in self.vars]
        for i in range(len(self.vars)):
            if list(self.domains[i]) != list(other.domains[other_map[i]]):
                return False

        for other_assignment in itertools.product(*other.domains):
            assignment = tuple(other_assignment[i] for i in other_map)
            if abs(self[assignment] - other[other_assignment]) > 1e-6:
                return False

        return True

    def __setitem__(self, assignment, value):
        if isinstance(assignment, dict):
            assignment = tuple(assignment[k] for k in self.vars)
        elif not isinstance(assignment, tuple):
            if len(self.vars) > 1:
                raise KeyError("Unable to understand the key")
            assignment = (assignment,)
        self.val[assignment] = value

    def __getitem__(self, assignment):
        if isinstance(assignment, dict):
            assignment = tuple(assignment[k] for k in self.vars)
        elif not isinstance(assignment, tuple):
            if len(self.vars) > 1:
                raise KeyError("Unable to understand the key")
            assignment = (assignment,)

        if assignment not in self.val:
            if self.default is None:
                raise KeyError("Factor value for assignment: %r is not set" % (assignment,))
            else:
                return self.default
        return self.val[assignment]

    def __iter__(self):
        for key in self.val.keys():
            yield key

    def __matmul__(self, other):
        for var in set(self.vars) & set(other.vars):
            if list(self.domains[self.vars.index(var)]) != list(other.domains[other.vars.index(var)]):
                raise ValueError("Domains of common variable %r do not match in both factors." % (var,))

        new_vars = sorted(set(self.vars) | set(other.vars))
        new_domains = []
        for var in new_vars:
            if var in self.vars:
                new_domains.append(self.domains[self.vars.index(var)])
            else:
                new_domains.append(other.domains[other.vars.index(var)])

        return Factor(new_vars, new_domains)

    def __mul__(self, other):
        new_factor = self @ other

        left_map = [new_factor.vars.index(var) for var in self.vars]
        right_map = [new_factor.vars.index(var) for var in other.vars]

        for assignment in itertools.product(*new_factor.domains):
            left_val = self[tuple(assignment[t] for t in left_map)] if left_map else 1  # else allows empty left
            right_val = other[tuple(assignment[t] for t in right_map)] if right_map else 1  # else allows empty right
            new_factor[assignment] = left_val * right_val

        return new_factor

    def __add__(self, other):
        new_factor = self @ other

        left_map = [new_factor.vars.index(var) for var in self.vars]
        right_map = [new_factor.vars.index(var) for var in other.vars]

        for assignment in itertools.product(*new_factor.domains):
            left_val = self[tuple(assignment[t] for t in left_map)] if left_map else 0  # else allows empty left
            right_val = other[tuple(assignment[t] for t in right_map)] if right_map else 0  # else allows empty right
            new_factor[assignment] = left_val + right_val

        return new_factor

    def dummy_marginalise(self, vars_to_marginalise):
        """
        Eliminates `vars_to_marginalise` without initializing resulting factor.
        i.e. just computes skeleton of the new factor.
        """
        vars_to_marginalise = set(vars_to_marginalise)
        for var in vars_to_marginalise:
            if var not in self.vars:
                raise ValueError("Variable %r not in factor" % (var,))

        new_vars_idx = [i for i, v in enumerate(self.vars) if v not in vars_to_marginalise]
        new_vars = [self.vars[i] for i in new_vars_idx]
        new_domains = [self.domains[i] for i in new_vars_idx]
        new_factor = Factor(new_vars, new_domains)
        return new_factor

    def marginalise(self, vars_to_marginalise):
        """Returns new marginalised factor where `vars_to_marginalise` are summed up."""
        new_factor = self.dummy_marginalise(vars_to_marginalise)
        new_vars_idx = [i for i, v in enumerate(self.vars) if v not in vars_to_marginalise]

        for assignment in itertools.product(*new_factor.domains):
            new_factor[assignment] = 0

        for assignment in itertools.product(*self.domains):
            new_assignment = tuple(assignment[i] for i in new_vars_idx)
            new_factor[new_assignment] += self.val[assignment]
        return new_factor

    def marginalise_in(self, vars_to_keep):
        return self.marginalise(set(self.vars) - set(vars_to_keep))

    def normalize(self):
        """normalizes the factor. Inplace operation. Returns nothing."""
        z = sum(self.val.values())
        for key in self.val:
            self.val[key] = self.val[key] / z
        return self

    def conditional_normalize(self, given):
        """Same as normalize, but conditional. That is, Sum_X P(X|given) = 1"""
        if not given:
            return self
        given = set(given)
        if given - set(self.vars):
            raise ValueError("`given` contains a variable not part of this factor")
        if given == set(self.vars):
            raise ValueError("`given` can't be all the variables of this factor")

        given_domains = []
        other_domains = []
        given_x = []
        other_x = []

        for x, d in zip(self.vars, self.domains):
            if x in given:
                given_domains.append(d)
                given_x.append(x)
            else:
                other_domains.append(d)
                other_x.append(x)

        for given_assignment in itertools.product(*given_domains):
            full_assignment = dict(zip(given_x, given_assignment))
            z = 0
            for other_assignment in itertools.product(*other_domains):
                full_assignment.update(zip(other_x, other_assignment))
                z += self[full_assignment]

            if abs(z) < 1e-6:
                continue

            for other_assignment in itertools.product(*other_domains):
                full_assignment.update(zip(other_x, other_assignment))
                self[full_assignment] /= z
        return self

    def evidence(self, evidence):
        """Returns new factor consistent with the evidence
        Args:
            evidence: A dictionary that maps variable to the observed value.
                e.g. {"coin1": "Heads"}

        Returns:
            new factor with same set of variables and same set of domains but
            factor values that are not consistent with the evidence are zeroed out.
        """
        if self.default is not None:
            raise NotImplementedError("Yet to implement for factor having default assignment")

        relevant_evidence = {}
        for v, e in evidence.items():
            if v in self.vars:
                idx = self.vars.index(v)
                if e not in self.domains[idx]:
                    raise ValueError("%r not in the domain of %r, which is %r" % (e, v, self.domains[idx]))
                relevant_evidence[idx] = e
        evidence = relevant_evidence

        if not evidence:
            return self
        new_factor = Factor(self.vars, self.domains)

        for assignment, value in self.val.items():
            for v, e in evidence.items():
                if assignment[v] != e:
                    new_factor[assignment] = 0
                    break
            else:
                new_factor[assignment] = value

        return new_factor

    def log_transform(self, inplace=False):
        if inplace:
            new_factor = self
        else:
            new_factor = Factor(self.vars, self.domains)

        for assignment in self.val:
            t = self.val[assignment]
            new_factor[assignment] = -np.inf if t == 0 else np.log(t)
        return new_factor

    @staticmethod
    def from_matlab(factor_dict, start_from_zero=True):
        """This method is used for course assignments.

        Factor saved in .mat format can be loaded by `scipy.io.loadmat` matrix and
        can be passed to this function to create Factor object.
        """
        var = factor_dict['var']
        if start_from_zero:
            var = var - 1

        card = factor_dict['card']
        if not isinstance(var, np.ndarray):
            var = [int(var)]
            card = [int(card)]
        else:
            var = var.astype(int).tolist()
            card = card.astype(int).tolist()

        f = Factor(var, card)
        for i, val in enumerate(factor_dict['val']):
            assignment = np.unravel_index(i, card, order='F')
            f[assignment] = val

        return f

    @staticmethod
    def from_mat_struct(struct, start_from_zero=True):
        var = struct.var
        if start_from_zero:
            var = var - 1

        card = struct.card
        if not isinstance(var, np.ndarray):
            var = [var]
            card = [card]
        else:
            var = var.tolist()
            card = card.tolist()

        f = Factor(var, card)
        for i, val in enumerate(struct.val):
            assignment = np.unravel_index(i, card, order='F')
            f[assignment] = val

        return f

    @staticmethod
    def from_pandas(df):
        df = df.sort_values(list(df.drop(columns=['value'])))
        domains = df.drop(columns=['value']).T.apply(lambda x: sorted(set(x)), axis=1)
        f = Factor(domains.index.tolist(), domains.tolist(), default=0)
        for x, y in zip(df[f.vars].values, df.value.values):
            f[tuple(x)] = y
        return f

    @staticmethod
    def from_pgm_cpd(cpd):
        import xarray as xr
        out_df = (
            xr.DataArray(cpd.values, dims=[str(x) for x in cpd.variables]).to_series().rename('value').reset_index()
        )
        out = Factor.from_pandas(out_df)
        return out

    def to_pgm_cpd(self):
        from pgmpy.factors.discrete.CPD import TabularCPD
        levels = list(-np.arange(len(self.domains) - 1) - 1)
        values = self.df.set_index(self.vars).unstack(level=levels).values
        card = [len(x) for x in self.domains]
        return TabularCPD(self.vars[0], card[0], values, evidence=self.vars[1:], evidence_card=card[1:])

    def to_pgm_cpd_sparse(self):
        fac = self.sparse_to_dense_cpd()
        return fac.to_pgm_cpd()

    def sparse_to_dense_cpd(self):
        import xarray as xr
        df = self.df
        cols = df.columns
        df.columns = [str(x) for x in cols]
        res = xr.DataArray.from_series(df.set_index(list(df)[:-1]).value)
        df = res.to_series().reset_index().fillna(0)
        df.columns = cols
        fac = Factor.from_pandas(df)
        return fac

    def to_pgm_jpd(self):
        from pgmpy.factors.discrete import JointProbabilityDistribution as JPD
        levels = list(-np.arange(len(self.domains) - 1) - 1)
        card = [len(x) for x in self.domains]
        return JPD(self.vars, card, self.df.value.values)

    @staticmethod
    def from_pandas_sparse(df, domains=None, default=None, complete=False):
        vars = list(df.dropcol('value'))
        df = df.sort_values(vars)
        if domains is None:
            domains = df.dropcol('value').T.apply(lambda x: sorted(set(x)), axis=1).list
        Factor2 = partial(Factor, init=default) if complete else partial(Factor, default=default)
        f = Factor2(vars, domains)
        for idx, val in df.set_index(f.vars).iterrows():
            f[idx] = val[0]
        return f


def log_prob_of_joint_assignment(factors, assignment):
    if isinstance(assignment, (list, tuple, np.ndarray)):
        assignment = {i: a for i, a in enumerate(assignment)}
    assert isinstance(assignment, dict)
    return np.sum([np.log(f[assignment]) for f in factors])


Factor.marginalize = Factor.marginalise
Factor.marginalize_in = Factor.marginalise_in

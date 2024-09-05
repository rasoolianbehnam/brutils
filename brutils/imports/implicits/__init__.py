import re
import atexit
import itertools
import operator
import subprocess
from collections import namedtuple
from functools import partial, reduce, wraps

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

import brutils.utility as ut
from brutils.utility import RegisterWithClass


def del_file(f):
    try:
        if f.startswith("hdfs://"):
            subprocess.getoutput(f"hdfs dfs -rm -r {f}")
            return
        elif f.startswth("s3://"):
            subprocess.getoutput(f"aws s3 rm --rercursive {f}")
            return
        subprocess.getoutput(f"rm -r {f}")
    except Exception as e:
        print(e)
        pass


def del_files():
    files = ut.get_resource("files_to_delete", list)
    for f in files:
        del_file(f)


atexit.register(del_files)


@RegisterWithClass(pd.DataFrame)
def renameCols(self, colsDict=None, **kwargs):
    if colsDict is None:
        colsDict = {}
    d = colsDict.copy()
    d.update({v: k for k, v in kwargs.items()})
    return self.rename(columns=d)


@RegisterWithClass(pd.DataFrame)
def FilterIndex(self, regex):
    return self[lambda x: x.index.str.match(regex)]


@RegisterWithClass(pd.DataFrame)
# @wraps(pd.DataFrame.merge)
def left_anti_join(self, b, **kwargs):
    on = kwargs.pop("on", None)
    if on is None:
        left_on, right_on = kwargs.pop("left_on", None), kwargs.pop("right_on", None)
        if left_on is None or right_on is None:
            on = list(set(self) & set(b))
            left_on, right_on = on, on
    else:
        left_on, right_on = on, on
    kwargs["how"] = "outer"
    kwargs["indicator"] = True
    return (
        self.merge(
            b[right_on].drop_duplicates(), left_on=left_on, right_on=right_on, **kwargs
        )
        .query('_merge == "left_only"')
        .drop("_merge", axis=1)
    )


@RegisterWithClass(pd.DataFrame)
def cartesian_join(self, b):
    return (
        self.assign(ababab=1).merge(b.assign(ababab=1), on="ababab").drop("ababab", 1)
    )


@RegisterWithClass(pd.DataFrame)
def left_semi_join(self, b, on=None):
    if on is None:
        on = list(set(self) & set(b))
    return self.merge(b[on].drop_duplicates(), on=on)


@RegisterWithClass(pd.DataFrame)
def SortValues(self, reverse=False):
    cols = list(self)
    if reverse:
        cols = cols[::-1]
    return self.sort_values(cols)


@RegisterWithClass(pd.DataFrame)
def sort(self, *b, asc=True):
    return self.sort_values(list(b), ascending=asc)


@RegisterWithClass(pd.DataFrame)
def Rename2(self, *args, **kwargs):
    if len(args):
        return self.rename(columns={v: k for k, v in args[0].items()})
    return self.rename(columns={v: k for k, v in kwargs.items()})


@RegisterWithClass(pd.DataFrame)
def dropcol(self, *args, ignore_nonexisting=True, **kwargs):
    if isinstance(args[0], list):
        args = args[0]
    if ignore_nonexisting:
        args = [x for x in args if x in self.columns]
    return self.drop(list(args), axis=1, **kwargs)


@RegisterWithClass(pd.DataFrame)
def evals(self, *exprs: str):
    exprs = itertools.chain.from_iterable([expr.split(";") for expr in exprs])
    out = self
    for expr in exprs:
        out = out.eval(expr)
    return out


@RegisterWithClass(pd.DataFrame)
def lower(self):
    out = self.copy()
    out.columns = [x.lower() for x in self.columns]
    return out


@RegisterWithClass(pd.DataFrame)
def set_value(self, values, set_index=True):
    index = None
    if set_index:
        index = self.index
    return pd.DataFrame(values, index=index, columns=self.columns)


@RegisterWithClass(pd.DataFrame)
def obj_to_cat(self, suffix="", cols=None):
    if cols is None:
        cols = list(self)
    out = self[cols].copy()
    for c in cols:
        s = out[c]
        if s.dtype == "object":
            out[c + suffix] = pd.Categorical(s)
    return out


@RegisterWithClass(pd.DataFrame)
def cat_to_code(self, suffix="", cols=None):
    if cols is None:
        cols = list(self)
    out = self[cols].copy()
    for c in cols:
        s = out[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            out[c + suffix] = s.cat.codes
    return out


@RegisterWithClass(pd.DataFrame)
def obj_to_code(self, suf1="", suf2="", cols=None):
    if cols is None:
        cols = list(self)
    return self.obj_to_cat(suf1, cols=cols).cat_to_code(suf2, cols=cols)


@RegisterWithClass(pd.DataFrame)
def coords(self, cols=None, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    if cols is None:
        cols = list(self)
    cols = list(set(cols) - set(drop_cols))
    d = self.obj_to_cat(cols=cols)
    coords = {"index": d.index.values}
    for c in cols:
        s = d[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            coords[c] = s.cat.categories.tolist()
        elif s.dtype != "float":
            coords[c] = sorted(s.unique())
    return coords


@RegisterWithClass(pd.DataFrame)
def Data(self, cols=None, dims="index"):
    import pymc3 as pm

    self = self.obj_to_code(cols=cols).reset_index()
    if cols is None:
        cols = list(self)
    Data = namedtuple("Data", cols)
    return Data(*[pm.Data(c, self[c].values, dims=dims) for c in cols])


@RegisterWithClass(pd.DataFrame)
def xarr(self, value_col):
    cols = list(set(self) - {value_col})
    return self.set_index(cols).to_xarray()


@RegisterWithClass(pd.DataFrame)
def truncated(tmp, beg=4, end=3):
    beg = list(range(beg))
    end = list(reversed([-x - 1 for x in range(end + 1)]))
    tmp = tmp.iloc[:, beg + end]
    tmp = tmp.Rename2(**{"...": tmp.columns[-len(end)]})
    tmp.iloc[:, -4] = "..."
    return tmp


@RegisterWithClass(pd.DataFrame)
def show_all_rows(self):
    from IPython.core.display import display

    n = pd.get_option("display.max_rows")
    pd.set_option("display.max_rows", None)
    display(self)
    pd.set_option("display.max_rows", n)


@RegisterWithClass(pd.DataFrame)
def sns(self, plot_name, **kwargs):
    import seaborn as sns

    getattr(sns, plot_name)(data=self, **kwargs)


@RegisterWithClass(pd.DataFrame)
def set_columns(self, cols):
    self = self.copy()
    self.columns = cols
    return self


@RegisterWithClass(pd.DataFrame)
def set_index2(self, idx):
    self = self.copy()
    self.index = idx
    return self


@RegisterWithClass(pd.DataFrame)
def apply2(self, fun):
    fun(self)
    return self


@RegisterWithClass(pd.DataFrame)
def apply3(self, s, inplace=False, **kwargs):
    if not inplace:
        self = self.copy()
    locals().update(kwargs)
    exec(s)
    return self


@RegisterWithClass(pd.DataFrame)
def stack2(self, index, level_1, value="value"):
    return (
        self.set_index(index)
        .stack()
        .reset_index()
        .Rename2(**{level_1: "level_1", value: 0})
    )


@RegisterWithClass(pd.DataFrame)
def groupByIndex(self, col_nums):
    cols = self.iloc[:, col_nums].columns.tolist()
    return self.groupby(cols)


@RegisterWithClass(pd.DataFrame)
def select(self, *cols):
    return self[list(cols)]


@RegisterWithClass(pd.DataFrame)
def selectExpression(self, *cols):
    splits = [col.split(" as ") for col in cols]
    cols = [x[0] for x in splits]
    renames = {k[0]: k[1] for k in splits if len(k) == 2}
    return self[list(cols)].rename(columns=renames)


@RegisterWithClass(pd.DataFrame)
def rselect(self, *cols):
    return pd.concat([self.filter(regex=col) for col in cols], axis=1)


@RegisterWithClass(pd.DataFrame)
def rdrop(self, *cols):
    cs = reduce(operator.add, [self.filter(regex=col).columns.tolist() for col in cols])
    return self.drop(columns=cs)


@RegisterWithClass(pd.DataFrame)
def filtex(self, *regex, **kwargs):
    tmp = self.iloc[:1]
    cols = reduce(
        operator.add, [list(tmp.filter(regex=r, **kwargs).columns) for r in regex]
    )
    return self[cols]


@RegisterWithClass(pd.DataFrame)
def filt(self, f):
    if callable(f):
        return self[f]
    return self.query(f)


@ut.RegisterWithClass(pd.DataFrame)
def dropcolregex(self, *patterns):
    def match(x):
        return any(re.match(pattern, x) for pattern in patterns)

    cols = [x for x in self.columns if not match(x)]
    return self[cols]


@RegisterWithClass(pd.Series)
def quantile_rank(self, k):
    return ((self.rank() / len(self)) * k).astype("int")


@RegisterWithClass(pd.Series)
def value_probs(self):
    out = self.value_counts()
    return out / out.sum()


@RegisterWithClass(pd.Series)
def normalize(self):
    return (self - self.mean()) / self.std()


@RegisterWithClass(pd.Series)
def toProb(self):
    return self / self.sum()


pd.DataFrame.dropcols = pd.DataFrame.dropcol
pd.DataFrame.rename2 = pd.DataFrame.Rename2
pd.DataFrame.Rename = pd.DataFrame.Rename2
pd.Series.cast = pd.Series.astype
pd.Series.list = property(pd.Series.tolist)
pd.Series.idx = property(lambda self: pd.DatetimeIndex(self))
pd.Series.distinct2 = pd.Series.drop_duplicates
pd.Series.sort = pd.Series.sort_values
pd.Series.to = pd.Series.astype
pd.DataFrame.distinct2 = pd.DataFrame.drop_duplicates
pd.DataFrame.to = pd.DataFrame.astype
pd.Index.list = property(pd.Index.to_list)

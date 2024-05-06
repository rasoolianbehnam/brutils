import hashlib
import operator
from functools import wraps, reduce

import brutils.utility as ut
import re
import subprocess
from datetime import datetime
from uuid import uuid4

from brutils import _Config
from brutils.utility import RegisterWithClass, read_dataframe
from pyspark.sql import DataFrame, GroupedData, Column, Window
import pyspark.sql.functions as F


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


# import atexit
# atexit.register(del_files)


@RegisterWithClass(DataFrame)
def assign(self, **kwargs):
    out = self
    for k, v in kwargs.items():
        if not isinstance(v, Column):
            v = F.lit(v)
        out = out.withColumn(k, v)
    return out


@RegisterWithClass(DataFrame)
def selex(self, *cols):
    return self.selectExpr(*cols)


@RegisterWithClass(DataFrame)
def groupBy2(self, *cols):
    cols = [F.expr(str(col)) for col in cols]
    return self.groupBy(*cols)


@RegisterWithClass(DataFrame)
def self_join(self, *args, **kwargs):
    return self.alias("a").join(self.alias("b"), *args, **kwargs)


@RegisterWithClass(GroupedData)
def Agg(self, *cols):
    cols = [F.expr(x) for x in cols]
    return self.agg(*cols)


@RegisterWithClass(GroupedData)
def probDist(self):
    return self.count().assign(p=F.col("count") / F.sum("count").over(Window.partitionBy()))


@RegisterWithClass(DataFrame)
def rename(self, d=None, **kwargs):
    if d is None:
        d = {}
    kwargs.update()
    out = self
    for k, v in kwargs.items():
        out = out.withColumnRenamed(str(v), str(k))
    return out


@RegisterWithClass(DataFrame)
def sums(self, *cols):
    return self.select([F.sum(c).alias(c) for c in cols])


@RegisterWithClass(DataFrame)
def lower(self):
    return self.select([c.lower() for c in self.columns])


def get_alias(k, v):
    alias = k + "_" + v.replace("*", "all")
    if " as " in v:
        v, alias = v.split(" as ", 1)
    alias = alias.strip()
    if alias == '':
        alias = v
    return v, alias


@RegisterWithClass(GroupedData)
def agg2(self, *args, agg=None, **kwargs):
    if agg is None:
        agg = []
    cols = prepare_args(args, kwargs)
    return self.agg(*cols, *agg)


@RegisterWithClass(GroupedData)
def summarize(self, **kwargs):
    aggs = [v.alias(k) for k, v in kwargs.items()]
    return self.agg(*aggs)


@RegisterWithClass(GroupedData)
def aggExpr(self, *args, agg=None, **kwargs):
    cols = [F.expr(x) for x in args]
    return self.agg(*cols)


@RegisterWithClass(DataFrame)
def select3(self, *args, **kwargs):
    cols = prepare_args(args, kwargs)
    return self.select(*cols)


@RegisterWithClass(DataFrame)
def select_dtype(self, *dtypes):
    dtypes = set(dtypes)
    if 'numeric' in dtypes:
        dtypes = dtypes.union({'bigint', 'double', 'int', 'float'})
    return self.select([x[0] for x in self.dtypes if x[1] in dtypes])


def prepare_args(args, kwargs):
    cols = []
    if len(args):
        d = args[0]
    else:
        d = kwargs
    for k, v in d.items():
        if isinstance(v, str):
            v, alias = get_alias(k, v)
            cols.append(getattr(F, k)(v).alias(alias))
        else:
            for v_ in v:
                v_, alias = get_alias(k, v_)
                cols.append(getattr(F, k)(v_).alias(alias))
    return cols


@RegisterWithClass(DataFrame)
def pandas(self, n: int = None, cache=True):
    if n is None:
        out = self
    else:
        out = self.limit(n)
    if cache:
        out = out.cache()
    return out.toPandas()


@RegisterWithClass(DataFrame)
def toJulia(self, n: int = None):
    if n is None:
        df = self.toPandas()
    else:
        df = self.limit(n).toPandas()
    df.to_julia()


@RegisterWithClass(DataFrame)
def cache2(self, path):
    self.write.mode('overwrite').parquet(path)
    return read_dataframe(path)


@RegisterWithClass(DataFrame)
def cache3(self: DataFrame, root: str = None):
    config = _Config()
    if root is None:
        root = config.root_by_device
    if not root.endswith("/"):
        root += "/"
    cols = "_".join([x.replace("_", "")[:2] for x in self.columns])
    d = datetime.now().strftime("%Y-%m-%d-%H-%M")
    path = f"{root}{cols}/{d}/{uuid4()}"
    print(path)
    ut.get_resource("files_to_delete", list).append(path)
    self.write.mode('overwrite').parquet(path)
    return read_dataframe(path)


@RegisterWithClass(DataFrame)
def cache4(self: DataFrame, force=False, root: str = None, partition_by=None, dry_run=False, name=''):
    path = self.cache4Path(root, prefix=name)
    if dry_run:
        print(path)
        return self
    print(path)
    writer = self.write
    if partition_by is not None:
        writer = writer.partitionBy(partition_by)
    if force:
        writer.mode('overwrite').parquet(path)
    else:
        try:
            writer.parquet(path)
        except:
            print("already exists")
    return read_dataframe(path)


@RegisterWithClass(DataFrame)
def cache4Path(self, root: str = None, prefix=''):
    config = _Config()
    if root is None:
        root = config.root_by_device
    if not root.endswith("/"):
        root += "/"
    a = prefix + "_" + self.encoded_query()[:15]
    today = datetime.strftime(datetime.today(), format="%Y-%m-%d")
    path = f"{root}tmp/cache4/{today}/{a}"
    return path


@RegisterWithClass(DataFrame)
def encoded_query(self):
    return str(hashlib.sha512(str(self.logical_query()).encode()).hexdigest())


@RegisterWithClass(DataFrame)
@wraps(cache4)
def cache5(self: DataFrame, **kwargs):
    return ut.run_in_thread(self.cache4, **kwargs)


@RegisterWithClass(DataFrame)
def logical_query(self):
    a = re.sub("#\d*", "", self._jdf.queryExecution().logical().toString())
    b = re.sub("#\d*", "", self._jdf.queryExecution().toString())
    return f"{a}\n{b}"


@RegisterWithClass(DataFrame)
def print_logical_query(self):
    print(self.logical_query())


@RegisterWithClass(DataFrame)
def select_regex(self, *patterns, strict=False):
    cols = self.colsRegex(patterns, strict)
    return self.select(cols)


@RegisterWithClass(DataFrame)
def drop_regex(self, *patterns, strict=False):
    cols = self.colsRegex(patterns, strict)
    return self.select([c for c in self.columns if c not in cols])


@RegisterWithClass(DataFrame)
def colsRegex(self, patterns, strict):
    cols = []
    for pattern in patterns:
        if strict:
            c = re.compile(pattern)
        else:
            c = re.compile('(?i)^.*' + pattern + '.*$')
        cols.extend([x for x in self.columns if c.match(x)])
    return cols


@RegisterWithClass(DataFrame)
def select_regex(self, *patterns, strict=False):
    cols = []
    for pattern in patterns:
        if strict:
            c = re.compile(pattern)
        else:
            c = re.compile('(?i)^.*' + pattern + '.*$')
        cols.extend([x for x in self.columns if c.match(x)])
    return self.select(cols)


@RegisterWithClass(DataFrame)
def or_filters(self, *filters):
    return self[reduce(operator.or_, filters)]


@RegisterWithClass(DataFrame)
def and_filters(self, *filters):
    return self[reduce(operator.and_, filters)]


def describe_(col, percentiles=None):
    if percentiles is None:
        percentiles = [.25, .5, .75]
    p = ','.join(["%.2f" % p for p in percentiles])
    x = F.expr(f"percentile({col}, array({p}))").alias("percentiles")
    return (
        F.count("*").alias("count"), F.avg(col).alias("mean"), F.stddev(col).alias("std"),
        F.min(col).alias("min"), F.max(col).alias("max"),
        *[x[i].alias(f"{int(p * 100)}%") for i, p in enumerate(percentiles)],
    )


@RegisterWithClass(DataFrame)
def describe2(self, subset=None, percentiles=None):
    from brutils.utility.spark_utils import union_by_col
    if subset is None:
        subset = self.columns
    res = [self.select(F.lit(col).alias("col"), *describe_(col, percentiles)) for col in subset]
    return union_by_col(res).select(res[0].columns)


@RegisterWithClass(DataFrame)
def self_join_lambda(self: DataFrame, f, on=None, **kwargs):
    if on is None:
        on = []
    other = f(self)
    if isinstance(on, str) or len(on):
        return self.join(other, on=on, **kwargs)
    return self.crossJoin(other)


@RegisterWithClass(GroupedData)
def describe(self, subset=None, percentiles=None):
    from brutils.utility.spark_utils import union_by_col
    if subset is None:
        subset = self.columns
    res = [self.agg(F.lit(col).alias("col"), *describe_(col, percentiles)) for col in subset]
    return union_by_col(res).select(res[0].columns)


@RegisterWithClass(GroupedData)
def Collect(self, *cols, col_name="collected"):
    return (
        self.agg(F.array_distinct(F.arrays_zip(*[F.collect_list(col) for col in cols])).alias(col_name))
    )


def expr2(col):
    out = [F.expr(x) for x in col.split(",") if len(x)]
    if len(out) == 1:
        return out[0]
    return out


@RegisterWithClass(DataFrame)
def filter_by_group(df, group, agg, filt, how="left_semi", join=None):
    """
    :param df:
    :param group:
    :param agg: can be an expression with variable agg in it
    :param filt:
    :param how:
    :param join:
    :return:
    """
    if join is None:
        join = group
    f = (
        df.groupby(group).agg(agg.alias('agg'))
            .filter(filt)
    )
    return df.join(f, on=join, how=how)


@RegisterWithClass(DataFrame)
def merge(self, df, on=None, **kwargs):
    if on is None:
        on = list(set([x.lower() for x in self.columns]) & set([x.lower() for x in df.columns]))
    return self.join(df, on=on, **kwargs)


@RegisterWithClass(DataFrame)
def filters(self, *args):
    str_filters = [f"({f})" for f in args if isinstance(f, str)]
    final_filters = [f for f in args if not isinstance(f, str)]
    if len(str_filters):
        final_filters.append(' and '.join(str_filters))
    out = self
    for f in final_filters:
        out = out.filter(f)
    return out


@RegisterWithClass(DataFrame)
def set_columns(df, cols):
    return df.rename(**dict(zip(cols, df.columns)))


@RegisterWithClass(DataFrame)
def filtex(self, pattern):
    import pandas as pd
    cols = pd.Series(self.columns)[lambda s: s.str.match(pattern)].tolist()
    return self[cols]


@RegisterWithClass(DataFrame)
def first_k(self, k, *cols, partitionBy=None):
    if partitionBy is None:
        partitionBy = []
    return self.assign(raghas123=F.dense_rank().over(Window.partitionBy(partitionBy).orderBy(*cols))) \
        .filter(F.col("raghas123") <= k).drop("raghas123")


@RegisterWithClass(DataFrame)
def prefix(self, prefix, *exceptions):
    prefix = prefix + "_"
    if len(exceptions) and isinstance(exceptions[0], list):
        exceptions = exceptions[0]
    rename_map = {prefix + c: c for c in set(self.columns) - set(exceptions)}
    return self.rename(**rename_map)


@RegisterWithClass(DataFrame)
def countDistribution(self, col):
    return self.groupby(col).agg2(count='* as n').groupby("n").count() \
        .assign(t=F.sum("count").over(Window.partitionBy()), p=F.col("count") / F.col("t")).drop("t").sort("n")


@RegisterWithClass(DataFrame)
def CheckManyToMany(df, col1=None, col2=None, extra_cols=None, aggs=None):
    if col1 is None:
        col1 = df.columns[0]
    if col2 is None:
        col2 = df.columns[1]
    if aggs is None:
        aggs = []
    aggs = [F.count] + aggs
    aggs_ = [f('a') for f in aggs]
    if extra_cols is None:
        extra_cols = []
    a = df.groupby([*extra_cols, col1]).agg(F.countDistinct(col2).alias("a")).filter("a>1").groupby(*extra_cols) \
        .agg(*aggs_) \
        .assign(name=f"one {col1} to many {col2}")
    b = df.groupby([*extra_cols, col2]).agg(F.countDistinct(col1).alias("a")).filter("a>1").groupby(*extra_cols) \
        .agg(*aggs_) \
        .assign(name=f"one {col2} to many {col1}")
    return a.union(b)


@RegisterWithClass(DataFrame)
def RemoveOneToMany(df, col1, col2=None):
    f = df.groupby(col1)
    if col2 is None:
        f = f.agg(F.count('*').alias("n"))
    else:
        f = f.agg(F.nunique(col2).alias("n"))
    f = f.filter("n==1")
    return df.merge(f, how="left_semi")


@RegisterWithClass(DataFrame)
def GetVenn(df1, df2, on, name1="a", name2="b", fancy=False):
    union = f"{name1} $\cup$ {name2}" if fancy else "union"
    intersection = f"{name1} $\cap$ {name2}" if fancy else "intersection"
    return (
        df1.select(on).distinct().withColumn("a", F.lit(1))
            .join(df2.select(on).distinct().withColumn("b", F.lit(1)), on=on, how="outer")
            .select(
            F.sum("a").alias(name1),
            F.sum("b").alias(name2),
            F.count('*').alias(union),
            F.sum(F.expr("cast(a*b as int)")).alias(intersection), )
            .assign(
            jaccard=F.col("intersection") / F.col("union"),
            p_intersect_a=F.col("intersection") / F.col(name1),
            p_intersect_b=F.col("intersection") / F.col(name2),
        )
    )


@RegisterWithClass(DataFrame)
def GetVennGrouped(df1, df2, on, groupby: list = None, name1="a", name2="b", fancy=False):
    if isinstance(on, str):
        on = [on]
    if groupby is None:
        groupby = []
    elif isinstance(groupby, str):
        groupby = [groupby]
    union = f"|{name1} $\cup$ {name2}|" if fancy else "union"
    intersection = f"|{name1} $\cap$ {name2}|" if fancy else "intersection"
    return (
        df1.select(on + groupby).distinct().withColumn("a", F.lit(1))
            .join(df2.select(on + groupby).distinct().withColumn("b", F.lit(1)), on=groupby+on, how="outer")
            .groupby(groupby)
            .agg(
            F.sum("a").cast('int').alias(name1),
            F.sum("b").cast('int').alias(name2),
            F.count('*').cast('int').alias(union),
            F.sum(F.expr(f"a*b")).cast('int').alias(intersection), )
            .assign(**{
                'jaccard':F.col(intersection) / F.col(union),
                f'p_intersect_{name1}':F.col(intersection) / F.col(name1),
                f'p_intersect_{name2}':F.col(intersection) / F.col(name2),
            }
        )
        .sort(groupby)
    )


@RegisterWithClass(DataFrame)
def MakeOneToOne(df, col1=None, col2=None, extra_cols=None, dropna=False):
    if extra_cols is None:
        extra_cols = []
    if col1 is None:
        col1 = df.columns[0]
    if col2 is None:
        col2 = df.columns[1]
    if dropna:
        df = df.dropna(subset=[col1, col2])
    a = df.groupby(*extra_cols, col1).agg(F.nunique(col2).alias("a")).filter("a>1")
    b = df.groupby(*extra_cols, col2).agg(F.nunique(col1).alias("a")).filter("a>1")
    return df.join(a, on=[*extra_cols, col1], how="left_anti").join(b, on=[*extra_cols, col2], how="left_anti")


@RegisterWithClass(DataFrame)
def sample_k(self, k):
    return self.assign(shamas134=F.rand()).first_k(k, "shamas134").drop("shamas134")


@RegisterWithClass(DataFrame)
def first_p(self, percentile, col, maximum=None):
    k = self.describe2([col], percentiles=[percentile]).pandas().iloc[0, -1]
    out = self.filter(F.col(col) <= k)
    if maximum:
        out = out.first_k(maximum, col)
    return out


@RegisterWithClass(DataFrame)
def top_p(self, percentile, col, maximum=None):
    k = self.describe2([col], percentiles=[1 - percentile]).pandas().iloc[0, -1]
    out = self.filter(F.col(col) >= k)
    if maximum:
        out = out.first_k(maximum, F.desc(col))
    return out


@RegisterWithClass(DataFrame)
def named_select(self, **kwargs):
    cols = [v.alias(k) for k, v in kwargs.items()]
    return self.select(cols)


@RegisterWithClass(DataFrame)
def explode(self, col, name=None):
    name = name or col
    cols = list(self.columns)
    cols.remove(col)
    return self.select(*cols, F.explode(col).alias(name))


@RegisterWithClass(F.Column)
def rlike_isin(self, l, compare_fun=F.Column.rlike):
    return reduce(operator.or_, [compare_fun(self, x) for x in l])


@RegisterWithClass(F.Column)
def map_from_dict(self, d, compare_fun=F.Column.__eq__, otherwise=None):
    if len(d) == 0:
        return self
    items = list(d.items())
    out = F.when(compare_fun(self, items[0][0]), F.lit(items[0][1]))
    for i in range(1, len(items)):
        out = out.when(compare_fun(self, items[i][0]), F.lit(items[i][1]))
    if otherwise is None:
        return out
    elif otherwise == 'self':
        return out.otherwise(self)
    else:
        return out.otherwise(otherwise)


def array_is_subset(a, b):
    return F.size(a) == F.size(F.array_intersect(a, b))


def percentile(col, percentiles):
    array = ', '.join([str(x) for x in percentiles])
    expr = f"percentile({col}, array({array}))"
    return F.expr(expr)


def isnull(c):
    return F.isnull(c) | F.isnan(c)


def nunique(*args):
    if isinstance(args[0], (list, tuple)):
        return F.countDistinct(*args[0])
    return F.countDistinct(*args)


def array_intersect_size(a, b):
    return F.size(F.array_intersect(a, b))


def array_union_size(a, b):
    return F.size(F.array_union(a, b))


def jaccard_ratio(a, b):
    return F.array_intersect_size(a, b) / F.array_union_size(a, b)


F.array_intersect_size = array_intersect_size
F.array_union_size = array_union_size
F.jaccard_ratio = jaccard_ratio
F.expr2 = expr2
F.nunique = nunique
F.unique = F.collect_set
F.set = F.collect_set
F.list = F.collect_list
F.assign_ids = lambda col: F.dense_rank().over(Window.orderBy(col)) - 1
F.cast = lambda col, tp: F.col(col).cast(tp)
F.acd = F.approx_count_distinct
F.isnull2 = isnull
F.percentile = percentile
F.array_is_subset = array_is_subset
DataFrame.query = DataFrame.filter
DataFrame.unique = DataFrame.distinct
DataFrame.groupByExpr = DataFrame.groupBy2
DataFrame.groupby2 = DataFrame.groupBy2
DataFrame.mutate = DataFrame.assign
DataFrame.group_by = DataFrame.groupby
DataFrame.group_by = DataFrame.groupby

from brutils.utility import RegisterWithClass
from snowflake.snowpark.table import Table
from snowflake.snowpark.dataframe import DataFrame
from snowflake.snowpark.relational_grouped_dataframe import (
    RelationalGroupedDataFrame as GroupedData,
)
from snowflake.snowpark.column import Column
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark import Window

import operator
from functools import reduce
import re

import snowflake
import pandas as pd


@RegisterWithClass(snowflake.snowpark.session.Session)
def list_tables(self):
    return pd.DataFrame(self.sql("show tables").collect())


@RegisterWithClass(Table, DataFrame)
def __repr__(self):
    cols = str([f"{k}::{v}" for k, v in self.dtypes])
    type_ = str(self.__class__.__name__)
    return f"{type_}{cols}".replace("'", "").replace("16777216", "")


@RegisterWithClass(Table, DataFrame)
def pandas(self, k=None):
    if k is not None:
        self = self.limit(k)
    return self.toPandas().lower()


@RegisterWithClass(Table, DataFrame)
def assign(self, **kwargs):
    out = self
    for k, v in kwargs.items():
        if isinstance(v, str):
            v = F.lit(v)
        out = out.withColumn(k, v)
    return out


@RegisterWithClass(Table, DataFrame)
def selex(self, *cols):
    return self.selectExpr(*cols)


@RegisterWithClass(Table, DataFrame)
def groupBy2(self, *cols):
    cols = [F.expr(str(col)) for col in cols]
    return self.groupBy(*cols)


@RegisterWithClass(Table, DataFrame)
def self_join(self, *args, **kwargs):
    return self.alias("a").join(self.alias("b"), *args, **kwargs)


@RegisterWithClass(GroupedData)
def Agg(self, *cols):
    cols = [F.expr(x) for x in cols]
    return self.agg(*cols)


@RegisterWithClass(GroupedData)
def probDist(self):
    return self.count().assign(
        p=F.col("count") / F.sum("count").over(Window.partitionBy())
    )


@RegisterWithClass(Table, DataFrame)
def rename(self, d=None, **kwargs):
    if d is None:
        d = {}
    kwargs.update()
    out = self
    for k, v in kwargs.items():
        out = out.withColumnRenamed(str(v), str(k))
    return out


@RegisterWithClass(Table, DataFrame)
def sums(self, *cols):
    return self.select([F.sum(c).alias(c) for c in cols])


@RegisterWithClass(Table, DataFrame)
def lower(self):
    return self.select([c.lower() for c in self.columns])


def get_alias(k, v):
    alias = k + "_" + v.replace("*", "all")
    if " as " in v:
        v, alias = v.split(" as ", 1)
    alias = alias.strip()
    if alias == "":
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


@RegisterWithClass(Table, DataFrame)
def select3(self, *args, **kwargs):
    cols = prepare_args(args, kwargs)
    return self.select(*cols)


@RegisterWithClass(Table, DataFrame)
def select_dtype(self, *dtypes):
    dtypes = set(dtypes)
    if "numeric" in dtypes:
        dtypes = dtypes.union({"bigint", "double", "int", "float"})
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


@RegisterWithClass(Table, DataFrame)
def logical_query(self):
    a = re.sub("#\d*", "", self._jdf.queryExecution().logical().toString())
    b = re.sub("#\d*", "", self._jdf.queryExecution().toString())
    return f"{a}\n{b}"


@RegisterWithClass(Table, DataFrame)
def print_logical_query(self):
    print(self.logical_query())


@RegisterWithClass(Table, DataFrame)
def select_regex(self, *patterns, strict=False):
    cols = self.colsRegex(patterns, strict)
    return self.select(cols)


@RegisterWithClass(Table, DataFrame)
def drop_regex(self, *patterns, strict=False):
    cols = self.colsRegex(patterns, strict)
    return self.select([c for c in self.columns if c not in cols])


@RegisterWithClass(Table, DataFrame)
def colsRegex(self, patterns, strict):
    cols = []
    for pattern in patterns:
        if strict:
            c = re.compile(pattern)
        else:
            c = re.compile("(?i)^.*" + pattern + ".*$")
        cols.extend([x for x in self.columns if c.match(x)])
    return cols


@RegisterWithClass(Table, DataFrame)
def select_regex(self, *patterns, strict=False):
    cols = []
    for pattern in patterns:
        if strict:
            c = re.compile(pattern)
        else:
            c = re.compile("(?i)^.*" + pattern + ".*$")
        cols.extend([x for x in self.columns if c.match(x)])
    return self.select(cols)


@RegisterWithClass(Table, DataFrame)
def or_filters(self, *filters):
    return self[reduce(operator.or_, filters)]


@RegisterWithClass(Table, DataFrame)
def and_filters(self, *filters):
    return self[reduce(operator.and_, filters)]


def describe_(col, percentiles=None):
    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75]
    p = ",".join(["%.2f" % p for p in percentiles])
    x = F.expr(f"percentile({col}, array({p}))").alias("percentiles")
    return (
        F.count("*").alias("count"),
        F.avg(col).alias("mean"),
        F.stddev(col).alias("std"),
        F.min(col).alias("min"),
        F.max(col).alias("max"),
        *[x[i].alias(f"{int(p * 100)}%") for i, p in enumerate(percentiles)],
    )


@RegisterWithClass(Table, DataFrame)
def describe2(self, subset=None, percentiles=None):
    from brutils.utility.spark_utils import union_by_col

    if subset is None:
        subset = self.columns
    res = [
        self.select(F.lit(col).alias("col"), *describe_(col, percentiles))
        for col in subset
    ]
    return union_by_col(res).select(res[0].columns)


@RegisterWithClass(Table, DataFrame)
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
    res = [
        self.agg(F.lit(col).alias("col"), *describe_(col, percentiles))
        for col in subset
    ]
    return union_by_col(res).select(res[0].columns)


@RegisterWithClass(GroupedData)
def Collect(self, *cols, col_name="collected"):
    return self.agg(
        F.array_distinct(F.arrays_zip(*[F.collect_set(col) for col in cols])).alias(
            col_name
        )
    )


def expr2(col):
    out = [F.expr(x) for x in col.split(",") if len(x)]
    if len(out) == 1:
        return out[0]
    return out


@RegisterWithClass(Table, DataFrame)
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
    f = df.groupBy(group).agg(agg.alias("agg")).filter(filt)
    return df.join(f, on=join, how=how)


@RegisterWithClass(Table, DataFrame)
def merge(self, df, on=None, **kwargs):
    if on is None:
        on = list(
            set([x.lower() for x in self.columns])
            & set([x.lower() for x in df.columns])
        )
    return self.join(df, on=on, **kwargs)


@RegisterWithClass(Table, DataFrame)
def filters(self, *args):
    str_filters = [f"({f})" for f in args if isinstance(f, str)]
    final_filters = [f for f in args if not isinstance(f, str)]
    if len(str_filters):
        final_filters.append(" and ".join(str_filters))
    out = self
    for f in final_filters:
        out = out.filter(f)
    return out


@RegisterWithClass(Table, DataFrame)
def set_columns(df, cols):
    return df.rename(**dict(zip(cols, df.columns)))


@RegisterWithClass(Table, DataFrame)
def filtex(self, pattern):
    import pandas as pd

    cols = pd.Series(self.columns)[lambda s: s.str.match(pattern)].tolist()
    return self[cols]


@RegisterWithClass(Table, DataFrame)
def first_k(self, k, *cols, partitionBy=None):
    if partitionBy is None:
        partitionBy = []
    return (
        self.assign(
            raghas123=F.dense_rank().over(
                Window.partitionBy(partitionBy).orderBy(*cols)
            )
        )
        .filter(F.col("raghas123") <= k)
        .drop("raghas123")
    )


@RegisterWithClass(Table, DataFrame)
def prefix(self, prefix, *exceptions):
    prefix = prefix + "_"
    if len(exceptions) and isinstance(exceptions[0], list):
        exceptions = exceptions[0]
    rename_map = {prefix + c: c for c in set(self.columns) - set(exceptions)}
    return self.rename(**rename_map)


@RegisterWithClass(Table, DataFrame)
def countDistribution(self, col):
    return (
        self.groupBy(col)
        .agg2(count="* as n")
        .groupBy("n")
        .count()
        .assign(
            t=F.sum("count").over(Window.partitionBy()), p=F.col("count") / F.col("t")
        )
        .drop("t")
        .sort("n")
    )


@RegisterWithClass(Table, DataFrame)
def CheckManyToMany(df, col1=None, col2=None, extra_cols=None, aggs=None):
    if col1 is None:
        col1 = df.columns[0]
    if col2 is None:
        col2 = df.columns[1]
    if aggs is None:
        aggs = []
    aggs = [F.count] + aggs
    aggs_ = [f("a") for f in aggs]
    if extra_cols is None:
        extra_cols = []
    a = (
        df.groupBy([*extra_cols, col1])
        .agg(F.countDistinct(col2).alias("a"))
        .filter("a>1")
        .groupBy(*extra_cols)
        .agg(*aggs_)
        .assign(name=f"one {col1} to many {col2}")
    )
    b = (
        df.groupBy([*extra_cols, col2])
        .agg(F.countDistinct(col1).alias("a"))
        .filter("a>1")
        .groupBy(*extra_cols)
        .agg(*aggs_)
        .assign(name=f"one {col2} to many {col1}")
    )
    return a.union(b)


@RegisterWithClass(Table, DataFrame)
def RemoveOneToMany(df, col1, col2=None):
    f = df.groupBy(col1)
    if col2 is None:
        f = f.agg(F.count("*").alias("n"))
    else:
        f = f.agg(F.nunique(col2).alias("n"))
    f = f.filter("n==1")
    return df.merge(f, how="left_semi")


@RegisterWithClass(Table, DataFrame)
def GetVenn(df1, df2, on, name1="a", name2="b", fancy=False):
    union = f"{name1} $\cup$ {name2}" if fancy else "union"
    intersection = f"{name1} $\cap$ {name2}" if fancy else "intersection"
    return (
        df1.select(on)
        .distinct()
        .withColumn("a", F.lit(1))
        .join(df2.select(on).distinct().withColumn("b", F.lit(1)), on=on, how="outer")
        .select(
            F.sum("a").alias(name1),
            F.sum("b").alias(name2),
            F.count("*").alias(union),
            F.sum(F.expr("cast(a*b as int)")).alias(intersection),
        )
        .assign(
            jaccard=F.col("intersection") / F.col("union"),
            p_intersect_a=F.col("intersection") / F.col(name1),
            p_intersect_b=F.col("intersection") / F.col(name2),
        )
    )


@RegisterWithClass(Table, DataFrame)
def GetVennGrouped(
    df1, df2, on, groupBy: list = None, name1="a", name2="b", fancy=False
):
    if isinstance(on, str):
        on = [on]
    if groupBy is None:
        groupBy = []
    elif isinstance(groupBy, str):
        groupBy = [groupBy]
    union = f"|{name1} $\cup$ {name2}|" if fancy else "union"
    intersection = f"|{name1} $\cap$ {name2}|" if fancy else "intersection"
    return (
        df1.select(on + groupBy)
        .distinct()
        .withColumn("a", F.lit(1))
        .join(
            df2.select(on + groupBy).distinct().withColumn("b", F.lit(1)),
            on=groupBy + on,
            how="outer",
        )
        .groupBy(groupBy)
        .agg(
            F.sum("a").cast("int").alias(name1),
            F.sum("b").cast("int").alias(name2),
            F.count("*").cast("int").alias(union),
            F.sum(F.expr(f"a*b")).cast("int").alias(intersection),
        )
        .assign(
            **{
                "jaccard": F.col(intersection) / F.col(union),
                f"p_intersect_{name1}": F.col(intersection) / F.col(name1),
                f"p_intersect_{name2}": F.col(intersection) / F.col(name2),
            }
        )
        .sort(groupBy)
    )


@RegisterWithClass(Table, DataFrame)
def MakeOneToOne(df, col1=None, col2=None, extra_cols=None, dropna=False):
    if extra_cols is None:
        extra_cols = []
    if col1 is None:
        col1 = df.columns[0]
    if col2 is None:
        col2 = df.columns[1]
    if dropna:
        df = df.dropna(subset=[col1, col2])
    a = df.groupBy(*extra_cols, col1).agg(F.nunique(col2).alias("a")).filter("a>1")
    b = df.groupBy(*extra_cols, col2).agg(F.nunique(col1).alias("a")).filter("a>1")
    return df.join(a, on=[*extra_cols, col1], how="left_anti").join(
        b, on=[*extra_cols, col2], how="left_anti"
    )


@RegisterWithClass(Table, DataFrame)
def sample_k(self, k):
    return self.assign(shamas134=F.rand()).first_k(k, "shamas134").drop("shamas134")


@RegisterWithClass(Table, DataFrame)
def first_p(self, percentile, col, maximum=None):
    k = self.describe2([col], percentiles=[percentile]).pandas().iloc[0, -1]
    out = self.filter(F.col(col) <= k)
    if maximum:
        out = out.first_k(maximum, col)
    return out


@RegisterWithClass(Table, DataFrame)
def top_p(self, percentile, col, maximum=None):
    k = self.describe2([col], percentiles=[1 - percentile]).pandas().iloc[0, -1]
    out = self.filter(F.col(col) >= k)
    if maximum:
        out = out.first_k(maximum, F.desc(col))
    return out


@RegisterWithClass(Table, DataFrame)
def named_select(self, **kwargs):
    cols = [v.alias(k) for k, v in kwargs.items()]
    return self.select(cols)


@RegisterWithClass(Table, DataFrame)
def explode(self, col, name=None):
    name = name or col
    cols = list(self.columns)
    cols.remove(col)
    return self.select(*cols, F.explode(col).alias(name))


@RegisterWithClass(Table, DataFrame)
def join2(self, other, left_on, right_on, *args, **kwargs):
    assert len(left_on) == len(right_on)
    left_on = [F.col(l) if isinstance(l, str) else l for l in left_on]
    right_on = [F.col(r) if isinstance(r, str) else r for r in right_on]
    on = reduce(lambda a, b: a | b, [l == r for l, r in zip(left_on, right_on)])
    return self.join(other, *args, on=on, **kwargs)


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
    elif otherwise == "self":
        return out.otherwise(self)
    else:
        return out.otherwise(otherwise)


@RegisterWithClass(Table, DataFrame)
def audit(self):
    r01 = (
        self[[F.nunique(col).alias(col) for col in self.columns]]
        .pandas()
        .iloc[0]
        .rename("cardinality")
    )
    r02 = (
        self[[F.sum(F.is_null(col).cast("int")).alias(col) for col in self.columns]]
        .pandas()
        .iloc[0]
        .rename("n_missing")
    )
    r03 = (
        self[
            [
                F.array_slice(F.collect_set(col), F.lit(0), F.lit(2)).alias(col)
                for col in self.columns
            ]
        ]
        .pandas()
        .iloc[0]
        .rename("sample_values")
    )
    return pd.concat(
        [
            pd.Series([x[1] for x in self.dtypes], name="data_types", index=r01.index),
            r01,
            r02,
            r03.map(eval),
        ],
        axis=1,
    )


@RegisterWithClass(Table, DataFrame)
def separate_columns_by_type(df):
    """
    Separate Snowpark DataFrame columns by type: numeric, categorical, and timestamp.

    Parameters:
    -----------
    df : snowflake.snowpark.dataframe.DataFrame
        Input Snowpark DataFrame

    Returns:
    --------
    dict : Dictionary with keys 'numeric', 'categorical', and 'timestamp'
           containing lists of column names for each type
    """
    from snowflake.snowpark.types import (
        IntegerType,
        LongType,
        ShortType,
        ByteType,
        FloatType,
        DoubleType,
        DecimalType,
        DateType,
        TimestampType,
        TimeType,
    )

    numeric_cols = []
    categorical_cols = []
    timestamp_cols = []

    for field in df.schema.fields:
        col_name = field.name
        col_type = type(field.datatype)

        if col_type in [
            IntegerType,
            LongType,
            ShortType,
            ByteType,
            FloatType,
            DoubleType,
            DecimalType,
        ]:
            numeric_cols.append(col_name)
        elif col_type in [DateType, TimestampType, TimeType]:
            timestamp_cols.append(col_name)
        else:
            categorical_cols.append(col_name)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "timestamp": timestamp_cols,
    }


def array_is_subset(a, b):
    return F.size(a) == F.size(F.array_intersect(a, b))


def percentile(col, percentiles):
    array = ", ".join([str(x) for x in percentiles])
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
F.list = F.collect_set
F.assign_ids = lambda col: F.dense_rank().over(Window.orderBy(col)) - 1
F.cast = lambda col, tp: F.col(col).cast(tp)
F.acd = F.approx_count_distinct
F.isnull2 = isnull
F.percentile = percentile
F.array_is_subset = array_is_subset


DataFrame.query = DataFrame.filter
DataFrame.unique = DataFrame.distinct
DataFrame.groupByExpr = DataFrame.groupBy2
DataFrame.groupBy2 = DataFrame.groupBy2
DataFrame.mutate = DataFrame.assign
DataFrame.group_by = DataFrame.groupBy
DataFrame.groupby = DataFrame.groupBy

Table.query = Table.filter
Table.unique = Table.distinct
Table.groupByExpr = Table.groupBy2
Table.groupBy2 = Table.groupBy2
Table.mutate = Table.assign
Table.group_by = Table.groupBy
Table.groupby = Table.groupBy

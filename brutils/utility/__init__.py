import math
import functools
import queue
import socket
import struct
import sys
from threading import Thread
from uuid import uuid4

import json
import os
import re
import subprocess
import types
from datetime import datetime, timedelta
from functools import reduce
from logging import getLogger
from typing import Dict, List, Any, Union, Tuple, Optional

import numpy as np
import pandas as pd
from pandas.errors import MergeError

import brutils
from IPython.display import HTML, Markdown

logger = getLogger()

GR_INDEX = 'gr'
AM_INDEX = 'am'
US_INDEX = 'us'
MP_INDEX = 'media_plans'
# DataFrameType = Union[pd.DataFrame, dd.DataFrame]
# DataFrame = Union[pd.DataFrame, dd.DataFrame, DataFrame]
DataFrameColumn = Union[str, List[str]]

default_rcparams = None


def get_gams_output(droot, m_basename='model'):
    list_file = f"{droot}/{m_basename}.lst"
    if os.path.exists(list_file):
        os.unlink(list_file)
    output = subprocess.getoutput(
        f"""cd {droot} && gams {m_basename}.gms > /dev/null && tail -n 30 {m_basename}.lst;""", )
    # if os.path.exists(list_file):
    #     os.unlink(list_file)
    return output


def get_variable_from_gams_log(log, var, decimals=2):
    c = re.compile(f"""\d+ VARIABLE {var}.*\n*(.*)\n""")
    values = c.findall(log)
    if len(values) == 1:
        values = values[0]
    elif len(values) == 0:
        raise RuntimeError(f"Variable {var} not found")
    else:
        raise RuntimeError(f"Variable {var} not found")
    str_value_pairs = [re.split("""\s+""", x.strip()) for x in values.split(',')]
    num_value_pairs = [(int(x), np.round(float(y), decimals)) for x, y in str_value_pairs]
    return dict(num_value_pairs)


def make_filter(x: pd.Series, contains: List = None, not_contains: List = None, case: bool = False):
    """
    Parameters
    -----------
    x: The series to search in
    contains: List of keywords that a row should contain
    not_contains: List of keywords that the a should not contain.
    case: ignore case ?

    Returns
    ---------
    The filter to use in the dataframe where the x comes from
    """
    contains = contains or []
    not_contains = not_contains or []
    out = []
    if len(contains):
        out.append(reduce(lambda t, y: t & y, (x.str.contains(y, case=case) for y in contains)))
    if len(not_contains):
        out.append(reduce(lambda t, y: t & y, (~x.str.contains(y, case=case) for y in not_contains)))
    return reduce(lambda t, y: t & y, out)


def get_from_acs_2017(var_names: Dict, target='state'):
    """
    Parameters
    -----------
    var_names
    target: should be in {"state", "zip code tabulation area"}
    :return:
    """
    import requests
    link_template = f"https://api.census.gov/data/2017/acs/acs5?get=NAME,%s&for={target}"
    link = link_template % ','.join(var_names.keys())
    print(link)
    response = requests.get(link)
    df = pd.DataFrame(json.loads(response.content))
    df.columns = df.iloc[0].values
    df = df.drop(0)
    df.columns = [df.columns[0], *var_names.values(), df.columns[-1]]
    df.index = df[target]
    return df.drop(["NAME", target], axis=1)


def weight_population_error(weights, gr_pop, us_pop, gr_us_map):
    gr_weighted_pop = weighted_population(weights, gr_pop)
    gr_us_map_with_pop = gr_weighted_pop.merge(gr_us_map, on='gr').merge(us_pop, on='us')
    merged = weights.merge(gr_us_map_with_pop, on='gr')
    return (merged['weighted_gr_pop'] - merged['us_pop']).abs() / merged['us_pop']


def weighted_population(weights, panel_pop):
    merged = panel_pop.merge(weights, on='gr')
    merged['weighted_gr_pop'] = merged['weights'] * merged['gr_pop']
    return merged


def weight_viewership_data(weights, gr_data, am_data, gr_am_map):
    gr_am_viwership = gr_data.merge(gr_am_map, on='gr').merge(am_data, on=['media_plans', 'am'])
    merged = weights.merge(gr_am_viwership, on='gr')
    merged['weighted_gr_reach'] = merged['weights'] * merged['gr_reach']
    merged['weighted_gr_impression'] = merged['weights'] * merged['gr_impression']
    pairs = merged.groupby(['media_plans', 'am']).sum()
    return pairs


def weight_viewership_error(weights, gr_data, am_data, gr_am_map, agg_fun=np.mean):
    pairs = weight_viewership_data(weights, gr_data, am_data, gr_am_map)
    reach_error = (pairs['weighted_gr_reach'] - pairs['am_reach']).abs() / pairs['am_reach']
    impression_error = (pairs['weighted_gr_impression'] - pairs['am_impression']).abs() / pairs['am_impression']
    return agg_fun(reach_error), agg_fun(impression_error)


def raise_exception(name: str = '') -> Any:
    def fun():
        raise ValueError(f"The variable {name} should be initiated")

    return fun


def dmd(*x: str):
    from IPython.core.display import display, Markdown as Md
    # noinspection PyTypeChecker
    return display(Md(' '.join(x)))


GR_DEMO_IDENTIFIERS = ['personAge', 'personGender', 'state', 'numPersonsInHousehold']
GR_TS_SHARED_COLS = ['personGender', 'personAge']
GR_US_SHARED_COLS = ['personAge', 'personGender', 'state']
gr_demo_identifiers = ['personAge', 'personGender', 'state', 'numPersonsInHousehold']
gr_viewership_cols = {
    "personReach": "gr_reach",
    "personImpressions": "gr_impression"
}
am_viewership_cols = {
    "amrld_person_reach": 'am_reach',
    "amrld_person_impressions": 'am_impression'
}


def clean_ax(*axes, sharex=True, sharey=True, same_scale=True):
    mns = []
    mxs = []
    for ax in axes:
        mnx, mxx = ax.get_xlim()
        mny, mxy = ax.get_ylim()
        mns.append(min(mnx, mny))
        mxs.append(max(mxx, mxy))
    mn = min(mns)
    mx = max(mxs)
    for ax in axes:
        if sharex:
            ax.set_xlim([mn, mx])
        if sharey:
            ax.set_ylim([mn, mx])
        ax.grid('on')
    return mn, mx


def get_dt_range(dt: str, dt_days: Tuple[int, int],
                 dt_hours: Tuple[int] = (0, 0),
                 fmt_in: str = "%Y%m%d", fmt_out: str = "%Y%m%d"
                 ):
    dt = datetime.strptime(dt, fmt_in)
    dt_beg = datetime.strftime(dt - timedelta(days=dt_days[0], hours=dt_hours[0]), fmt_out)
    dt_end = datetime.strftime(dt + timedelta(days=dt_days[1], hours=dt_hours[1]), fmt_out)
    return dt_beg, dt_end


def old_read_dataframe(x, fmt='parquet', spark=None, options=None, *args, **kwargs):
    if options is None:
        options = {}
    import brutils.utility.spark_utils as spu
    spark = spark or spu.spark
    if isinstance(x, str):
        try:
            return spark.sql(x)
        except Exception:
            x = x.replace("s3://", "s3a://")
            if fmt == 'parquet':
                return spark.read.parquet(x, *args, **kwargs)
            elif fmt == 'csv':
                return spark.read.options(header='true', **options).csv(x, *args, **kwargs)
    return x


@functools.lru_cache(maxsize=1000)
def read_dataframe(x, fmt='parquet', spark=None, kwargs=None, **options):
    if kwargs is None:
        kwargs = {}
    import brutils.utility.spark_utils as spu
    spark = spark or spu.spark
    if isinstance(x, tuple):
        x = list(x)
    if isinstance(x, (str, list)):
        try:
            return spark.sql(x)
        except Exception:
            x = _correct_s3_paths(x)
            read = spark.read.format(fmt).options(**options)
            return read.load(x, **kwargs)
    return x


def _correct_s3_paths(x):
    if isinstance(x, list):
        return [_correct_s3_path(l) for l in x]
    return _correct_s3_path(x)


def _correct_s3_path(x):
    x = x.replace("s3://", "s3a://")
    return x


def cartesian_product(left, right, **kwargs):
    try:
        return pd.merge(left, right, **kwargs, how="outer")
    except MergeError:
        return cartesian_product_strict(left, right, **kwargs)


def cartesian_product_strict(left, right, extra_on=None, **kwargs):
    if extra_on is None:
        extra_on = []
    key = 'key2345'
    return pd.merge(left.assign(**{key: 1}), right.assign(**{key: 1}), on=[key, *extra_on], **kwargs) \
        .drop(key, 1)


def get_outlier_filter(x, ratio):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x > q3 + ratio * iqr) | (x < q1 - ratio * iqr)


def get_count_ratios(x):
    return x / x.sum()


def timestamp(level='day') -> str:
    if level == 'day':
        return datetime.strftime(datetime.now(), '%Y%m%d')
    elif level == 'hour':
        return datetime.strftime(datetime.now(), '%Y%m%d:%H')
    elif level == 'min':
        return datetime.strftime(datetime.now(), '%Y%m%d:%H%M')
    else:
        return datetime.strftime(datetime.now(), '%Y%m%d:%H%M%S')


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def plot_style(reset=False):
    import matplotlib
    from cycler import cycler
    global default_rcparams
    default_rcparams = matplotlib.rcParams if default_rcparams is None else default_rcparams
    if reset:
        matplotlib.rcParams.update(default_rcparams)
        return
    s = {
        "lines.linewidth": 2.0,
        "axes.edgecolor": "#bcbcbc",
        "patch.linewidth": 0.5,
        "legend.fancybox": True,
        "axes.prop_cycle": cycler('color', [
            "#348ABD",
            "#A60628",
            "#7A68A6",
            "#467821",
            "#CF4457",
            "#188487",
            "#E24A33"
        ]),
        "axes.facecolor": "#eeeeee",
        "axes.labelsize": "large",
        "axes.grid": True,
        "grid.linestyle": 'dashed',
        "grid.color": 'black',
        "grid.alpha": .2,
        "patch.edgecolor": "#eeeeee",
        "axes.titlesize": "x-large",
        "svg.fonttype": "path",

        "figure.figsize": (15, 9)
    }
    matplotlib.rcParams.update(s)


class DEVICE_OR_PERSON:
    def __init__(self, person: str = "persons", device: str = "devices"):
        self.person = person
        self.device = device


def unimplemented(varname=''):
    def fun():
        raise ValueError(f"Missing parameter {varname}")

    return fun


def CheckColsIn(cols, df):
    cols = {x.lower() for x in cols}
    columns = {x.lower() for x in df.columns}
    return cols & columns


def CheckColsInRegex(col, df, flags=re.IGNORECASE):
    return [x for x in df.columns if re.match(col, x, flags)]


def GetAxisMinMax(ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        xMin, xMax = plt.xlim()
        yMin, yMax = plt.ylim()
    else:
        xMin, xMax = ax.get_xlim()
        yMin, yMax = ax.get_ylim()
    return min(xMin, yMin), max(xMax, yMax)


@functools.wraps(np.random.choice)
def choice(a, size=1, **kwargs):
    n = len(a)
    replace = kwargs.pop('replace', False)
    if size > n:
        replace = True
    return np.random.choice(a, size=size, replace=replace, **kwargs)


class RegisterWithClass:
    def __init__(self, *clss: type):
        self.clss: Tuple[type] = clss

    def __call__(self, fun):
        name = fun.__name__
        for cls in self.clss:
            setattr(cls, name, fun)


def ip2int(addr):
    return struct.unpack("!I", socket.inet_aton(addr))[0]


def int2ip(addr):
    return socket.inet_ntoa(struct.pack("!I", addr))


def custom_matplotlib_style(figsize=(15, 15)):
    from cycler import cycler
    s = {
        "lines.linewidth": 2.0,
        "axes.edgecolor": "#bcbcbc",
        "patch.linewidth": 0.5,
        "legend.fancybox": True,
        "axes.prop_cycle": cycler('color', [
            "#348ABD",
            "#A60628",
            "#7A68A6",
            "#467821",
            "#CF4457",
            "#188487",
            "#E24A33"
        ]),
        "axes.facecolor": "#eeeeee",
        "axes.labelsize": "large",
        "axes.grid": True,
        "grid.linestyle": 'dashed',
        "grid.color": 'black',
        "grid.alpha": .2,
        "patch.edgecolor": "#eeeeee",
        "axes.titlesize": "x-large",
        "svg.fonttype": "path",
        'figure.figsize': figsize,
    }

    import matplotlib
    matplotlib.rcParams.update(s)


def FindAllMatching(l, r):
    return pd.Series(l)[lambda x: x.str.match(r)].tolist()


def CamelCaseToUnderscore(sourceString): return re.sub('([a-z]+)([A-Z])', r'\1_\2', sourceString).lower()


def create_dataframe(df: pd.DataFrame, verbose=1, **to_parquet_kwargs):
    from brutils import config
    import brutils.utility.spark_utils as spu
    from py4j.protocol import Py4JJavaError
    try:
        out = spu.spark.createDataFrame(df)
        out.pandas(2)
        return out
    except Py4JJavaError:
        if verbose:
            print("An error occured to to python mismatch")
        pass
    s = df.sample(min(20, len(df) // 2), random_state=3).applymap(hash).sum().sum() + sum(hash(x) for x in df.columns)
    today = datetime.strftime(datetime.today(), format="%Y-%m-%d")
    path = config.root + f"tmp/create_df/{today}/tmp{s}.parquet"
    if verbose:
        print(path)
    try:
        return read_dataframe(path)
    except:
        if verbose:
            print("not there")
        df.to_parquet(path, index=False, **to_parquet_kwargs)
        return read_dataframe(path)


def create_dataframe_from_tmp(df: pd.DataFrame):
    s = df.sample(min(20, len(df) // 2), random_state=3).applymap(hash).sum().sum()
    today = datetime.strftime(datetime.today(), format="%Y-%m-%d")
    dir = f"/tmp/create_df/{today}/"
    os.makedirs(dir, exist_ok=True)
    path = f"{dir}tmp{s}.parquet"
    print(path)
    df.to_parquet(path, index=False)
    return read_dataframe("file://" + path)


def str_to_pd(s, delimiter="\t", columns=None, drop_first=False):
    res = pd.DataFrame([x.split(delimiter) for x in s.strip().split("\n")])
    for col in res:
        res[col] = res[col].str.strip()
    if columns is not None:
        res.columns = columns
    else:
        res.columns = res.iloc[0]
    if drop_first:
        res.drop(0, inplace=True)
    return res


def pandas_df_to_excel(df_list: List[pd.DataFrame], sheet_list: List[str], file_name: str, **kwargs):
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0, startcol=0, **kwargs)
    writer.save()


def pandas_df_to_excel_dict(df_dict, file_name: str, **kwargs):
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    for sheet, dataframe in df_dict.items():
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0, startcol=0, **kwargs)
    writer.save()


def get_resource(k, default=None):
    try:
        return getattr(brutils, k)
    except AttributeError as e:
        v = default
        if callable(default):
            v = default()
        setattr(brutils, k, v)
        return v


class Future:
    def __init__(self, q):
        self.done = None
        self.value = None
        self.q = q

    @property
    def result(self):
        if self.done:
            return self.value
        value = self.q.get()
        self.done = True
        self.value = value
        return self.value

    @property
    def isdone(self):
        return not self.q.empty()


def run_in_thread(fun, *args, **kwargs):
    q = queue.Queue()

    def mardas():
        q.put(fun(*args, **kwargs))

    t = Thread(target=mardas)
    t.start()
    return Future(q)


def run_in_thread2(fun, *args, **kwargs):
    q = queue.Queue()

    def mardas():
        try:
            q.put(fun(*args, **kwargs))
        except Exception as e:
            q.put(e)

    t = Thread(target=mardas)
    t.start()
    return q


def convert_sparse_matrix_to_sparse_tensor(X):
    import tensorflow as tf
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def df_to_csr_matrix(x, ncols=None, nrows=None, rows_map=None, cols_map=None):
    from scipy import sparse
    if 'value' not in x:
        x['value'] = 1
    # assume rows start from 0
    if rows_map is None:
        rows_map = {x: i for i, x in enumerate(sorted(set(x['rows'])))}
    if cols_map is None:
        cols_map = {x: i for i, x in enumerate(sorted(set(x['rows'])))}
    if rows_map:
        x['rows'] = x.rows.map(rows_map)
    if cols_map:
        x['cols'] = x.cols.map(cols_map)
    x = x.dropna(subset=['rows', 'cols'])
    x[['rows', 'cols']] = x[['rows', 'cols']].astype('int')
    rows, cols = x.rows.values, x.cols.values
    if ncols is None:
        ncols = cols.max() + 1
    if nrows is None:
        nrows = rows.max() + 1
    return sparse.csr_matrix((x.value.values, (rows, cols)), shape=(nrows, ncols))


def download_google_drive_image(link, name):
    import tempfile
    import subprocess
    l = get_google_drive_download_link(link)
    d = f"{tempfile.mkdtemp()}/{name}"
    subprocess.getoutput(f"wget -q -O {d} '{l}'")
    return d


def get_google_drive_download_link(link):
    *_, a, b = link.split("/")
    l = f"https://drive.google.com/uc?id={a}&{'&'.join(b.split('&')[1:])}"
    print(l)
    return l


def display_google_drive_image(link):
    from IPython.display import Image
    import uuid
    d = download_google_drive_image(link, str(uuid.uuid4())[:8] + ".png")
    return Image(d)


def sql_list_from_iter(l):
    if isinstance(l, str):
        return f"('{l}')"
    return "(" + ', '.join([f"'{x}'" for x in l]) + ")"


def reverse_dict(d):
    return {v: k for k, v in d.items()}


def set_gpu_num(n):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n)


def formatter(s, i_format="{:,}", f_format="{:4,.2f}"):
    if isinstance(s, (int, np.int32, np.int64)):
        return i_format.format(s)
    if isinstance(s, (float, np.float32, np.float64)):
        return f_format.format(s)
    if isinstance(s, str):
        return s
    if isinstance(s, bool):
        return s
    if s is None or np.isnan(s):
        return s
    return s


def detect_dates(l):
    s = pd.Series(l)
    return (
        pd.to_datetime(s.str.extract("(\d\d+)[-/_]*(\d+)[_/-]*(\d+)").sum(1).astype('str').str[:-2])
            .rename("date").to_frame().assign(value=l)
    )


def byte_to_human_readable(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def camelCaseToTitle(s):
    return re.sub("([A-Z_]|\d+)", " \\1", s).replace("_", "").title()


def snakeToTitle(s):
    return s.replace("_", " ").title()


def figSize(*size):
    import matplotlib.pyplot as plt
    plt.figure(figsize=size)


def find_attributes(module, pattern, partial=True):
    if partial:
        pattern = f"^.*{pattern}.*$"
    return [x for x in dir(module) if re.match(pattern, x)]


@functools.wraps(pd.date_range)
def date_range(*args, **kwargs):
    delim = kwargs.pop("delim", '_')
    date = pd.date_range(*args, **kwargs)
    date = pd.DataFrame(date, columns=['date'])
    date['date_str'] = date.date.dt.strftime(f"%Y{delim}%m{delim}%d")
    return date


def split(lst, nSplits=None, splitSize=None):
    if nSplits is None and splitSize is None:
        raise ValueError("Both nSplits and splitSize are None")
    elif nSplits is not None and splitSize is not None:
        raise ValueError("Both nSplits and splitSize are not None")
    elif nSplits is not None:
        splitSize = len(lst) // nSplits
    else:
        nSplits = len(lst) // splitSize

    N = len(lst)
    e = 0
    for i in range(nSplits):
        s = i * splitSize
        e = min((i + 1) * splitSize, N)
        yield lst[s:e]
    if e < N:
        yield lst[e:]


def add_to_system_path(*paths):
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)


def Print(text):
    display(Markdown(text))


def html(text, font_size='16px', font_family='Arial', color='black', background_color='white'):
    html = f'''<p style="font-size: {font_size}; font-family: {font_family}; color: {color}; background-color: {background_color};">{text}</p>'''
    display(HTML(html))


def pyarray_to_series(a, name=None):
    if a.type == 'string':
        dtype = 'string[pyarrow]'
    elif a.type == 'large_string':
        dtype = 'large_string[pyarrow]'
    else:
        try:
            dtype = a.type.to_pandas_dtype()
            dtype = re.findall('\'numpy\.(.*?)_*\'', str(dtype))[0]
            dtype = dtype + '[pyarrow]'
            # .replace("numpy.", "")+'[pyarrow]'
        except:
            dtype = 'object'
    return pd.Series(a, dtype=dtype, name=name)



def pyarrow_to_pandas(df):
    return pd.concat([pyarray_to_series(df[c], c) for c in df.column_names], axis=1)


pd_to_sparse_matrix = df_to_csr_matrix
openDf = read_dataframe

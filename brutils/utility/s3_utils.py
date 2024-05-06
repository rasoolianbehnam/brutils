# version 3
import os
import shutil
from os.path import splitext, basename, join
from subprocess import getoutput
from typing import Dict, Any, Type, Callable, Tuple
from uuid import uuid4

import boto3
import dask.dataframe as dd
import joblib
import pandas as pd
from dask import compute


use_spark = True
verbose = 0

import brutils.utility as ut


# def to_s3(df, s3_target, overwrite=False):
#     try:
#         df.to_parquet(s3_target)
#     except:
#         d = f"/tmp/{str(uuid4())}/"
#         f = "data.parquet"
#         os.makedirs(d, exist_ok=True)
#         try:
#             df.to_parquet(d + f)
#             resource = s3.resource()
#             s3.upload_file(d + f, s3_target, resource, overwrite=overwrite)
#         finally:
#             shutil.rmtree(d)

def S3a(s3_path):
    return s3_path.replace("s3://", "s3a://")


def save_parquet(obj, file_path: str):
    import brutils.spark_utils as spu
    if spu.spark is not None and use_spark:
        spu.spark.createDataFrame(obj).write.parquet(file_path)
    else:
        obj.to_parquet(file_path)


def read_parquet(file_path: str):
    import brutils.spark_utils as spu
    if verbose:
        print("running read_parquet")
    if spu.spark is not None and use_spark:
        out = ut.read_dataframe(file_path)
        return out
    if is_dir(file_path):
        out = dd.read_parquet(join(file_path, "*parquet"))
    else:
        out = pd.read_parquet(file_path)
    if not file_path.startswith("s3://"):
        return compute(out)[0]
    return out


SaveFun = Callable[[Any, str], None]
ReadFun = Callable[[str], Any]

save_meta_data: Dict[Type, Tuple[SaveFun, str]] = {
    pd.DataFrame: (save_parquet, '.parquet'),
    dd.DataFrame: (save_parquet, '.parquet')
}

read_meta_data: Dict[str, ReadFun] = {
    '.parquet': read_parquet
}


def is_dir(s3_or_local_path: str):
    if s3_or_local_path.startswith("s3://"):
        res = list_s3_files(s3_or_local_path)
        return len(res) > 0
        # return len(list(s3.get_dirs(s3_or_local_path, resource))) > 0
    else:
        return os.path.isdir(s3_or_local_path)


def save_files_to_s3(root: str, data: Dict[str, Any]):
    for name, obj in data.items():
        save_to_s3(obj, os.path.join(root, name))


def list_s3_files(s3_files_path):
    s3_files_path = s3_files_path.replace("s3a://", "s3://")
    s3_files_path = clean_s3_file_path(s3_files_path) + "/"
    res = getoutput(f"aws s3 ls {s3_files_path}").split("\n")
    return [y for y in [x.split(' ')[-1] for x in res] if len(y)]


def read_files_from_s3(s3_files_path: str):
    files = list_s3_files(s3_files_path)
    out = {}
    for file in files:
        file = clean_s3_file_path(join(s3_files_path, file))
        if verbose:
            print(file)
        obj = read_from_s3(file)
        name = splitext(basename(file))[0]
        out[name] = obj
        if verbose:
            print("************************")
    return out


def save_to_s3(obj: Any, file_path: str):
    file_path = clean_s3_file_path(file_path)
    fun, ext = save_meta_data.get(type(obj), (joblib.dump, '.pkl'))
    try:
        fun(obj, file_path + ext)
        return
    except (FileNotFoundError, AttributeError, ImportError) as e:
        print(e)
        print("Writing via local ... ", end='')
        write_to_s3_via_local(obj, file_path, fun)
        print("done!")


def write_to_s3_via_local(obj, file_path, fun):
    os.makedirs("/tmp/weight/", exist_ok=True)
    tmp_name = f"/tmp/weight/{str(uuid4())}"
    try:
        fun(obj, tmp_name)
        copy_to_s3(file_path, tmp_name)
    finally:
        remove_file_or_dir(tmp_name)
    assert not os.path.exists(tmp_name)


def copy_to_s3(s3_path, local_path):
    recursive = ''
    if is_dir(local_path):
        recursive = '--recursive'
    os.system(f"aws s3 cp {recursive} {local_path} {s3_path}")


def clean_s3_file_path(s3_path):
    while s3_path.endswith('/'):
        s3_path = s3_path[:-1]
    return s3_path


def read_from_s3(s3_path: str, extension: str = None):
    s3_path = clean_s3_file_path(s3_path)
    extension = extension or splitext(s3_path)[1]
    fun = read_meta_data.get(extension, joblib.load)
    try:
        out = fun(s3_path)
    except (FileNotFoundError, ImportError, TypeError) as e:
        if verbose:
            print(e)
        out = read_from_s3_via_local_new(s3_path, fun)
    return out


def read_from_s3_via_local_new(s3_path, read_function):
    os.makedirs("/tmp/weight/", exist_ok=True)
    local_path = f"/tmp/weight/{str(uuid4())}"
    try:
        copy_from_s3(s3_path, local_path)
        out = read_function(local_path)
    finally:
        remove_file_or_dir(local_path)
    assert not os.path.exists(local_path)
    return out


def read_from_s3_via_local(s3_path, extension, read_function):
    os.makedirs("/tmp/weight/", exist_ok=True)
    local_path = f"/tmp/weight/{str(uuid4())}{extension}"
    try:
        copy_from_s3(s3_path, local_path)
        out = read_function(local_path)
    finally:
        remove_file_or_dir(local_path)
    assert not os.path.exists(local_path)
    return out


def remove_file_or_dir(local_path):
    if os.path.exists(local_path):
        if is_dir(local_path):
            shutil.rmtree(local_path)
        else:
            os.remove(local_path)


def copy_from_s3(s3_path, local_path):
    recursive = ""
    if is_dir(s3_path):
        recursive = "--recursive"
    os.system(f"aws s3 cp {recursive} {s3_path} {local_path}")



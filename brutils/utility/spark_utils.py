import functools
import getpass
import os
import numpy as np
import sys
import requests
from functools import wraps
from typing import Dict, Any, Optional, Tuple, Union, List

from pyspark import __version__ as spark_version
import conda_pack
from py4j.protocol import Py4JError
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import DataFrame, functions as F, types as T, GroupedData
from pyspark.sql.types import StringType
from pyspark.sql.utils import ParseException, AnalysisException
import brutils.utility as ut

DataFrameOrPath = Union[str, DataFrame]

SNOWFLAKE_SOURCE_NAME = "net.snowflake.spark.snowflake"
VERTICA_PASSWORD = ""

spark: Optional[SparkSession] = None
sql_context: Optional[SQLContext] = None
sc: Optional[SparkContext] = None
intp = None
conf = None

SF_OPTIONS = {
    "sfURL": "",
    "sfAccount": "",
    "sfUser": "",
    "sfPassword": "",
    "sfDatabase": "",
    "sfSchema": "PUBLIC",
    "sfWarehouse": "",
    "sfRole": "",
    'query': {'warehouse': '', 'role': 'readonly'}
}
MSSQL_OPTIONS = {'drivername': "com.microsoft.sqlserver.jdbc.SQLServerDriver",
                 'username': '',
                 'password': '',
                 'host': ''}

SPARK_CONF = {
    'spark.submit.deployMode': 'client',
    'spark.dynamicAllocation.maxExecutors': 150,
    # 'spark.driver.memoryOverhead': '10gb',
    # 'spark.executor.memoryOverhead': '10gb'
    # 'spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT': '1',
    # "spark.hadoop.fs.s3a.access.key": aws_creds['key'],
    # "spark.hadoop.fs.s3a.secret.key": aws_creds['secret'],
    # "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
}


def download_file(url, root=''):
    local_filename = file_name_from_url(url, root)
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return local_filename


def file_name_from_url(url, root):
    local_filename = os.path.join(root, url.split('/')[-1])
    return local_filename


def get_maven_link(x):
    x = x.replace(':jar:', ':')
    xa = x.split(':')
    x0 = xa[0].replace('.', '/')
    x12 = xa[1] + '-' + xa[2] + '.jar'
    return os.path.join(
        "https://repo1.maven.org/maven2/", x0, xa[1], xa[2], x12
    )


def download_jar_from_maven_link(link, root, overwrite):
    os.makedirs(root, exist_ok=True)
    file_name = file_name_from_url(link, root)
    if not os.path.exists(file_name) or overwrite:
        return download_file(link, root=root)
    else:
        print(f"file {file_name} already exits.")
        return file_name


def get_jar_names_from_packages(packages, jar_path, protocol='file://'):
    jars = [os.path.basename(get_maven_link(package)) for package in packages]
    jar_root = os.path.abspath(jar_path)
    return [protocol + os.path.join(jar_root, jar) for jar in jars]


def download_jar_files_from_packages(packages, root, overwrite):
    return [download_jar_from_maven_link(get_maven_link(package), root, overwrite) for package in packages]


def start_spark(master: str = 'yarn',
                name: str = 'brasoolian-spark',
                spark_conf: Dict[str, Any] = None,
                spark_home: Optional[str] = "/usr/lib/spark",
                pyspark_python=None,
                pyspark_driver_python=sys.executable,
                packages=None,  # usually used with yarn
                hive: bool = False
                ) -> Tuple[SparkContext, SparkSession, SQLContext]:
    # conf
    spark_conf = get_spark_conf_and_set_variables(
        master, spark_conf, spark_home, pyspark_python, pyspark_driver_python, packages
    )

    conf: SparkConf = SparkConf()
    conf.setMaster(master)
    conf.setAppName(name)
    # conf.setExecutorEnv('PYTHONPATH','./spark_zip/spark/bin/python')
    # conf.setExecutorEnv('PYSPARK_PYTHON','./spark_zip/spark/bin/python')

    for k, v in spark_conf.items():
        conf.set(k, v)
    spark_out_builder: SparkSession.builder = SparkSession.builder.config(conf=conf)
    if hive:
        spark_out_builder = spark_out_builder.enableHiveSupport()
    spark_out = spark_out_builder.getOrCreate()
    sc_out = spark_out.sparkContext
    sql_context_out = SQLContext(spark_out.sparkContext)
    return sc_out, spark_out, sql_context_out


def get_spark_conf_and_set_variables(master, spark_conf, spark_home, pyspark_python, pyspark_driver_python, packages):
    spark_conf = get_spark_conf(packages, spark_conf)
    # environments
    if spark_home is None and 'SPARK_HOME' in os.environ:
        del os.environ['SPARK_HOME']
    if pyspark_python is None:
        if master == 'yarn':
            pyspark_python = 'python3'
        else:
            pyspark_python = sys.executable
    os.environ['SPARK_HOME'] = spark_home
    os.environ['PYSPARK_PYTHON'] = pyspark_python
    os.environ['PYSPARK_DRIVER_PYTHON'] = pyspark_driver_python
    return spark_conf


def get_spark_conf(packages, spark_conf):
    if spark_conf is None:
        spark_conf = {}
    spark_conf = {**SPARK_CONF, **spark_conf}
    if packages is not None:
        jars = download_jar_files_from_packages(packages, "/tmp/jars/", overwrite=False)
        jar_files = ','.join("file://" + jar for jar in jars)
        spark_conf['spark.jars'] = spark_conf.get('spark.jars', '') + ',' + jar_files
        if int(spark_version.split('.')[0]) < 3:
            spark_conf['spark.yarn.jars'] = spark_conf.get('spark.jars', '') + "," + jar_files
    print(f"using {spark_conf}")
    return spark_conf


def start_spark_with_conda_env(
        conda_env: str, spark_conf=None,
        path_to_zip_or_tar: str = None, overwrite=False, **kwargs
) -> Tuple[SparkContext, SparkSession, SQLContext]:
    pyspark_python, spark_conf = get_spark_conf_with_conda_env(conda_env, spark_conf, path_to_zip_or_tar, overwrite)
    out = start_spark(spark_conf=spark_conf, pyspark_python=pyspark_python, **kwargs)  # for tests
    return out


def get_spark_conf_with_conda_env(conda_env, spark_conf, path_to_zip_or_tar=None, overwrite=False):
    if spark_conf is None:
        spark_conf = {}
    if path_to_zip_or_tar is None:
        path_to_zip_or_tar = f"/tmp/{conda_env}.tar.gz"
    if overwrite and os.path.exists(path_to_zip_or_tar):
        os.unlink(path_to_zip_or_tar)
    if not os.path.exists(path_to_zip_or_tar):
        print(f"Packing environment {conda_env} at {path_to_zip_or_tar}")
        conda_pack.pack(conda_env,
                        ignore_editable_packages=True,
                        ignore_missing_files=True,
                        output=path_to_zip_or_tar)
    if ".tar.gz" in path_to_zip_or_tar:  # made from conda-pack
        pyspark_python = f"./{conda_env}_zip/bin/python"
    elif '.zip' in path_to_zip_or_tar:  # normal zip
        pyspark_python = f"./{conda_env}_zip/{conda_env}/bin/python"
    else:
        raise ValueError("You should either provide a tar or a zip")
    spark_conf.update({
        'spark.yarn.dist.archives': f"{path_to_zip_or_tar}#{conda_env}_zip",
        'spark.yarn.appMasterEnv.PYSPARK_PYTHON': pyspark_python,
    })
    return pyspark_python, spark_conf


def load_snowflake_table_or_query(
        *, query: str = None, table: str = None,
        sf_options: Dict[str, Any] = None, sql_context_=None
) -> DataFrame:
    if query is None and table is None:
        raise ValueError("Both query and table cannot be None")
    elif query is not None and table is not None:
        raise ValueError("Either query or table must be None")
    query_or_table = "query" if query is not None else "dbtable"
    if sf_options is None:
        sf_options = SF_OPTIONS
    if table is not None and 'query' in sf_options:
        del sf_options['query']
    if sql_context_ is None:
        global sql_context
        sql_context_ = sql_context
    sf_viewership = sql_context_.read.format(SNOWFLAKE_SOURCE_NAME) \
        .options(**sf_options).option(query_or_table, query or table).load()
    return sf_viewership.lower()


def load_mssql_table_or_query(
        *, query: str = None, table: str = None,
        mssql_options: Dict[str, Any] = None, sql_context_=None
) -> DataFrame:
    if query is None and table is None:
        raise ValueError("Both query and table cannot be None")
    elif query is not None and table is not None:
        raise ValueError("Either query or table must be None")
    query_or_table = "query" if query is not None else "dbtable"
    if mssql_options is None:
        mssql_options = MSSQL_OPTIONS
    if sql_context_ is None:
        global sql_context
        sql_context_ = sql_context

    if query is not None:
        key, value = "query", query
    else:
        key, value = "dbtable", table
    server_addr = mssql_options['host']
    server_name = f"jdbc:sqlserver://{server_addr}"
    url = server_name + ";"
    try:
        out = sql_context_.read \
            .format("com.microsoft.sqlserver.jdbc.spark") \
            .option("driver", mssql_options["drivername"]) \
            .option("url", url) \
            .option("user", mssql_options['username']) \
            .option("password", mssql_options['password']) \
            .option(key, value) \
            .load()
        return out
    except ValueError as error:
        print("Connector write failed", error)


def load_vertica_table_or_query(
        *, query: str = None, table: str = None,
        vertica_options: Dict[str, Any] = None, sql_context_=None
) -> DataFrame:
    if query is not None:
        key, value = "query", query
    else:
        key, value = "dbtable", table
    if sql_context_ is None:
        global sql_context
        sql_context_ = sql_context

    global VERTICA_PASSWORD
    if not VERTICA_PASSWORD:
        VERTICA_PASSWORD = getpass.getpass("Please Enter LDAP password for brasoolian: ")
    try:
        out = sql_context_.read.format("jdbc").options(
            url="jdbc:vertica://url:port/database",
            driver="com.vertica.jdbc.Driver",
            user="brasoolian",
            password=VERTICA_PASSWORD,
        ).option(key, value).load()
        return out
    except ValueError as error:
        print("Connector write failed", error)


@wraps(start_spark)
def initialize(*args, **kwargs):
    global spark, sql_context, sc
    if sc is None:
        sc, spark, sql_context = start_spark(*args, **kwargs)
    return sc, spark, sql_context


@wraps(start_spark_with_conda_env)
def initialize_with_conda_env(*args, **kwargs):
    global spark, sql_context, sc
    if sc is None:
        sc, spark, sql_context = start_spark_with_conda_env(*args, **kwargs)
    return sc, spark, sql_context


def stop():
    global spark, sc, sql_context
    spark.stop()
    spark, sc, sql_context = None, None, None


def __initialize__(master,
                   spark_home,
                   spark_conf=None,
                   packages=None,
                   jar_packages=None,
                   name="brasoolian-scala",
                   path_to_zip_or_tar="",
                   pyspark_python=sys.executable,
                   ):
    """
    :param master:
    :param spark_home:
    :param spark_conf:
    :param packages:
    :param name:
    :param path_to_zip_or_tar: should have base name the same as environment name
    :return:
    """
    if jar_packages is None:
        jar_packages = []
    from spylon_kernel import get_scala_interpreter, register_ipython_magics
    from IPython import get_ipython
    global spark, sc, sql_context, intp
    if spark_conf is None:
        spark_conf = SPARK_CONF
    if packages is None:
        packages = []
    if master == 'yarn' and path_to_zip_or_tar:
        conda_env = path_to_zip_or_tar.split("/")[-1].split(".")[0]
        pyspark_python, conf_ = get_spark_conf_with_conda_env(
            conda_env=conda_env,
            path_to_zip_or_tar=path_to_zip_or_tar,
            spark_conf=spark_conf,
            overwrite=False
        )
        spark_conf.update(conf_)
    spark_conf = get_spark_conf(jar_packages, spark_conf)
    global conf
    conf = get_spark_conf_and_set_variables(
        master=master,
        spark_conf=spark_conf,
        spark_home=spark_home,
        pyspark_python=pyspark_python,
        pyspark_driver_python=sys.executable,
        packages=[],
    )

    register_ipython_magics()
    code = f'''import brutils.spark_utils as spu

launcher.master = "{master}"
launcher.conf.spark.app.name = "{name}"
for k, v in spu.conf.items():
    launcher.conf.set(k, v)
'''

    if len(packages) > 0:
        code += f"launcher.packages = {packages}\n"
    print(code)
    get_ipython().run_cell_magic(
        'init_spark', '', code)

    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()
    sql_context = SQLContext(sc)
    try:
        intp = sc._jvm.scala.tools.nsc.interpreter.IMain();
        intp.interpret("""System.getProperty("java.version")""")
    except Py4JError:
        pass
    intp = get_scala_interpreter()
    return intp


@functools.wraps(F.udf)
def udf(f, returnType=StringType(), **kwargs):
    f.__module__ = '__main__'
    f.__globals__.update(kwargs)
    out = F.udf(f, returnType)
    return out


def PrepareAgg(*columns):
    return F.arrays_zip(*[F.collect_list(col) for col in columns])


@functools.wraps(F.udf)
def AggUdf(f, returnType=StringType(), **kwargs):
    def u(*x):
        return udf(f, returnType, **kwargs)(PrepareAgg(*x))

    return u


def union_by_col(dataframes: List[DataFrame], fill_na=np.nan) -> DataFrame:
    # Create a list of all the column names and sort them
    cols = set()
    for df in dataframes:
        for x in df.columns:
            cols.add(x)
    cols = sorted(cols)

    # Create a dictionary with all the dataframes
    dfs: Dict[str, DataFrame] = {}
    for i, d in enumerate(dataframes):
        new_name = 'df' + str(i)  # New name for the key, the dataframe is the value
        dfs[new_name] = d
        # Loop through all column names. Add the missing columns to the dataframe (with value 0)
        for x in cols:
            if x not in d.columns:
                dfs[new_name] = dfs[new_name].withColumn(x, F.lit(fill_na))
        dfs[new_name] = dfs[new_name].select(cols)  # Use 'select' to get the columns sorted

    # Now put it al together with a loop (union)
    result: DataFrame = dfs['df0']  # Take the first dataframe, add the others to it
    dfs_to_add = set(dfs.keys())  # List of all the dataframes in the dictionary
    dfs_to_add.remove('df0')  # Remove the first one, because it is already in the result
    for x in dfs_to_add:
        result = result.union(dfs[x])
    return result


def CacheDf(df: ut.pd.DataFrame, dfPath: str, overwrite=False):
    if overwrite:
        df.write.mode('overwrite').parquet(dfPath)
    try:
        return ut.read_dataframe(dfPath)
    except AnalysisException:
        df.write.parquet(dfPath)
        return ut.read_dataframe(dfPath)


def SaltedGroupBy(df, groupByFields, aggFields, n, sameName=False):
    return df.withColumn("salt", (F.rand() * n).cast('integer')) \
        .groupBy("salt", *groupByFields) \
        .agg(*[f(*col).alias(f"{i}") if isinstance(col, list) else f(col).alias(f"{i}") for i, (f, col, *_) in
               enumerate(aggFields)]) \
        .groupBy(*groupByFields) \
        .agg(*[f(f"{i}").alias(alias[0]) if len(alias) else f(f"{i}").alias(col) if sameName else f(f"{i}")
               for i, (f, col, *alias) in enumerate(aggFields)])


def get_credentials_for_buckets(buckets, c):
    spark_aws_conf = {}
    for bucket in buckets:
        if bucket == "" or bucket is None:
            spark_aws_conf.update({
                f"fs.s3a.bucket.aws.credentials.provider": "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider",
                f"fs.s3a.bucket.access.key": c.access_key,
                f"fs.s3a.bucket.secret.key": c.secret_key,
                f"fs.s3a.bucket.session.token": c.token,
            })
        else:
            spark_aws_conf.update({
                f"fs.s3a.bucket.{bucket}.aws.credentials.provider": "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider",
                f"fs.s3a.bucket.{bucket}.access.key": c.access_key,
                f"fs.s3a.bucket.{bucket}.secret.key": c.secret_key,
                f"fs.s3a.bucket.{bucket}.session.token": c.token,
            })
    return spark_aws_conf


def set_credentials_for_buckets(buckets, c):
    for bucket in buckets:
        spark.sparkContext._jsc.hadoopConfiguration().set(f"fs.s3a.bucket.{bucket}.aws.credentials.provider",
                                                          "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider")
        spark.sparkContext._jsc.hadoopConfiguration().set(f"fs.s3a.bucket.{bucket}.access.key", c.access_key)
        spark.sparkContext._jsc.hadoopConfiguration().set(f"fs.s3a.bucket.{bucket}.secret.key", c.secret_key)
        spark.sparkContext._jsc.hadoopConfiguration().set(f"fs.s3a.bucket.{bucket}.session.token", c.token)


def clear_cache():
    global spark
    if spark is not None:
        spark.sql("clear cache")
    else:
        print("Spark session is not created")

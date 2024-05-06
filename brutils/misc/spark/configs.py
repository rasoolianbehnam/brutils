# app9980 spark
import brutils.utility.spark_utils as spu
import os
import socket


def connect_app9980_old(aws_creds, executor_instances=150, master="yarn"):
    jars_ = (
        "org.antlr_antlr4-4.7.jar,org.antlr_antlr4-runtime-4.7.jar,org.antlr_antlr-runtime-3.5.2.jar,"
        "org.antlr_ST4-4.0.8.jar,org.abego.treelayout_org.abego.treelayout.core-1.0.3.jar,"
        "org.glassfish_javax.json-1.0.4.jar,com.ibm.icu_icu4j-58.2.jar,io.delta_delta-core_2.11-0.4.0.jar"
    ).split(",")
    jars = []
    for file in jars_:
        if os.path.exists(path_to_jars + file):
            jars.append(file)
        else:
            print("jar not found")
    jars = [
        "file://{path_to_jars}" + jar for jar in jars
    ]

    packages = [
        "com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.5",
        "com.amazonaws:aws-java-sdk:1.7.4",
        "net.snowflake:snowflake-jdbc:3.12.17",
        "net.snowflake:spark-snowflake_2.11:2.8.4-spark_2.4",
        "org.apache.hadoop:hadoop-aws:2.7.7",
        "io.delta:delta-core_2.11:0.4.0",
        "org.tensorflow:spark-tensorflow-connector_2.11:1.15.0",
        "com.microsoft.azure:spark-mssql-connector:1.0.2",
        'com.mysql:mysql-connector-java:5.0.4'

    ]
    spu.initialize(
        master=master,
        packages=packages,
        spark_conf={
            # "spark.driver.userClassPathFirst": "false",
            "spark.executor.instances": executor_instances,
            "spark.executor.memory": "7gb",
            "spark.executor.memoryOverhead": "4gb",
            "spark.driver.memory": "40g",
            "spark.driver.maxResultSize": "20g",
            "spark.executor.cores": 4,
            "spark.sql.session.timeZone": "America/Detroit",
            "spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT": "1",
            'spark.jars': ','.join(jars),
            **aws_creds,
            # "spark.yarn.queue": "videology_bi_science",
        },
        name="brasoolian-spark01",
        spark_home="/opt/spark-2.4.4/",
        hive=True,
    )
    return spu.spark


def connect_app9980(aws_creds, path_to_zip_or_tar=None, executor_instances=150, master="yarn"):
    jars_ = (
        "org.antlr_antlr4-4.7.jar,org.antlr_antlr4-runtime-4.7.jar,org.antlr_antlr-runtime-3.5.2.jar",
        "org.antlr_ST4-4.0.8.jar,org.abego.treelayout_org.abego.treelayout.core-1.0.3.jar",
        "org.glassfish_javax.json-1.0.4.jar,com.ibm.icu_icu4j-58.2.jar,io.delta_delta-core_2.11-0.4.0.jar",
        "mssql-jdbc-9.3.1.jre8-preview.jar",
    )
    jars = []
    for file in jars_:
        if os.path.exists(path_to_jars + file):
            jars.append(file)
        else:
            print("jar not found")
    jars = [
        "file://{path_to_jars}" + jar for jar in jars
    ]

    packages = [
        "com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.5",
        "com.amazonaws:aws-java-sdk:1.7.4",
        "net.snowflake:snowflake-jdbc:3.12.17",
        "net.snowflake:spark-snowflake_2.11:2.8.4-spark_2.4",
        "org.apache.hadoop:hadoop-aws:2.7.7",
        "io.delta:delta-core_2.11:0.4.0",
        "org.tensorflow:spark-tensorflow-connector_2.11:1.15.0",
        "com.microsoft.azure:spark-mssql-connector:1.0.2",
        'mysql:mysql-connector-java:8.0.13',
        'com.vertica.jdbc:vertica-jdbc:24.1.0-0',
    ]
    if path_to_zip_or_tar is None:
        spu.initialize(
            master=master,
            packages=packages,
            spark_conf={
                # "spark.driver.userClassPathFirst": "false",
                "spark.executor.instances": executor_instances,
                "spark.executor.memory": "7gb",
                "spark.executor.memoryOverhead": "4gb",
                "spark.driver.memory": "40g",
                "spark.driver.maxResultSize": "20g",
                "spark.executor.cores": 4,
                "spark.sql.session.timeZone": "America/Detroit",
                "spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT": "1",
                'spark.jars': ','.join(jars),
                **aws_creds,
                # "spark.yarn.queue": "videology_bi_science",
            },
            name="brasoolian-spark01",
            spark_home="/opt/spark-2.4.4/",
            hive=True,
        )
    else:
        # get conda env from zip or tar file name
        conda_env = path_to_zip_or_tar.split("/")[-1].split(".")[0]
        spu.initialize_with_conda_env(
            master=master,
            conda_env=conda_env,
            path_to_zip_or_tar=path_to_zip_or_tar,
            packages=packages,
            spark_conf={
                # "spark.driver.userClassPathFirst": "false",
                "spark.executor.instances": executor_instances,
                "spark.executor.memory": "7gb",
                "spark.executor.memoryOverhead": "4gb",
                "spark.driver.memory": "40g",
                "spark.driver.maxResultSize": "20g",
                "spark.executor.cores": 4,
                "spark.sql.session.timeZone": "America/Detroit",
                "spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT": "1",
                'spark.jars': ','.join(jars),
                **aws_creds,
                # "spark.yarn.queue": "videology_bi_science",
            },
            name="brasoolian-spark01",
            spark_home="/opt/spark-2.4.4/",
            hive=True,
        )
    return spu.spark


def connect_hadoop_orig(master='yarn', spark_conf=None):
    if spark_conf is None:
        spark_conf = {}
    packages = [
        'net.snowflake:snowflake-jdbc:3.13.6',
        'net.snowflake:snowflake-ingest-sdk:0.10.3',
        'net.snowflake:spark-snowflake_2.12:2.9.1-spark_3.1',
        "com.microsoft.azure:spark-mssql-connector:1.0.2",
        'mysql:mysql-connector-java:8.0.13',
        'com.vertica.jdbc:vertica-jdbc:24.1.0-0',
    ]
    spu.initialize(
        master=master,
        spark_home="/usr/lib/spark/",
        # conda_env='p7d',
        # path_to_zip_or_tar='../../dependencies/conda_env/p7d.tar.gz',
        spark_conf={
            'spark.executor.memoryOverhead': '4gb',
            'spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT': '1',
            **spark_conf,
            # **spark_aws_conf,
        },
        packages=packages,
        name="brasoolian-spark-1",
    )
    return spu.spark


def connect_hadoop_orig_conda(conda_env_path, master='yarn', spark_conf=None):
    if spark_conf is None:
        spark_conf = {}
    packages = [
        'net.snowflake:snowflake-jdbc:3.13.6',
        'net.snowflake:snowflake-ingest-sdk:0.10.3',
        'net.snowflake:spark-snowflake_2.12:2.9.1-spark_3.1',
    ]
    conda_env = conda_env_path.split("/")[-1].split(".")[0]
    spu.initialize_with_conda_env(
        master=master,
        spark_home="/usr/lib/spark/",
        conda_env=conda_env,
        path_to_zip_or_tar=conda_env_path,
        spark_conf={
            'spark.executor.memoryOverhead': '4gb',
            'spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT': '1',
            **spark_conf,
            # **spark_aws_conf,
        },
        packages=packages,
        name="brasoolian-spark-1",
    )
    return spu.spark


def auto_connect(*args, **kwargs):
    host_dict = {
        'app9980.atl1.turn.com': connect_app9980,
        'ip-10-231-57-41': connect_hadoop_orig,
    }
    hostname = socket.gethostname()
    return host_dict[hostname](*args, **kwargs)


def find_all_tables(pattern):
    for db in spu.spark.sql("show databases").pandas().databaseName:
        d = spu.spark.sql(f"show tables in {db}").pandas()[lambda d: d.tableName.str.match(f"(?i)^{pattern}.*$")]
        if len(d):
            display(d)



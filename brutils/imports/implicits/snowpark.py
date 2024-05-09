import brutils.utility as ut
from snowflake.snowpark.table import Table
from snowflake.snowpark.dataframe import DataFrame


@ut.RegisterWithClass(Table, DataFrame)
def __repr__(self):
    return f"DataFrame{str([x.lower() for x in self.columns])}"


@ut.RegisterWithClass(Table, DataFrame)
def pandas(self, k=None):
    if k is not None:
        self = self.limit(k)
    return self.toPandas()
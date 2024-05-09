import snowflake.connector
import pandas as pd

class Snowflake:
    def __init__(self, credentials):
        self.cnx = snowflake.connector.connect(**credentials)

    def query(self, query):
        return pd.read_sql_query(query, self.cnx)

    def release(self):
        self.cns.close()
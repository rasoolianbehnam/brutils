import brutils.utility.spark_utils as spu
from typing import Dict, Any

import getpass
class Vertica:
    def __init__(self, user, sql=spu.sql_context, password=None):
        self.user = user
        self.sql = sql

        if password is None:
            self.VERTICA_PASSWORD = getpass.getpass(f"Please Enter LDAP password for {self.user}: ")
        else:
            self.VERTICA_PASSWORD = password

    def __call__(
            self, *, query: str = None, table: str = None,
            vertica_options: Dict[str, Any] = None,
    ):
        if query is not None:
            key, value = "query", query
        else:
            key, value = "dbtable", table
        try:
            out = self.sql.read.format("jdbc").options(
                url="jdbc:vertica://vertica-main-primary.service.iad1.data.nxxn.io:5433/verticaprod01",
                driver="com.vertica.jdbc.Driver",
                user=self.user,
                password=self.VERTICA_PASSWORD,
            ).option(key, value).load()
            return out
        except ValueError as error:
            print("Connector write failed", error)


class VerticaQuery:
    def __init__(self, user, password=None):
        self.vertica = Vertica(user, password=password)
        self.__dict__['tables'] = {}

    def __setattr__(self, key, value):
        # self.__dict__.setdefault('tables', {})[key] = value
        if isinstance(value, str) and value.strip().startswith("select"):
            self.__dict__['tables'][key] = value
        else:
            self.__dict__[key] = value

    def __getattr__(self, key):
        if key in self.__dict__['tables']:
            return self(self.__dict__['tables'][key])
        return self.__dict__[key]

    def __dir__(self):
        return self.__dict__['tables']

    def _tables(self):
        ts = ",\n".join([f"{name} as (\n{query})" for name, query in self.tables.items()])
        return f"with {ts}" if len(ts.strip()) else ""

    def __call__(self, query):
        full_query = self.get_query_string(query)
        return self.vertica(query=full_query)

    def get_query_string(self, query):
        tables = self._tables()
        full_query = f"{tables}\n{query}"
        return full_query

    def table(self, table):
        res = self(f"select * from {table}")
        return res

    def query(self, query):
        return self(query)

    def copy(self):
        out = self.__class__(self.vertica.user, self.vertica.VERTICA_PASSWORD)
        for k, v in self.__dict__.items():
            if k != 'tables':
                out.__dict__[k] = v
        for k, v in self.__dict__['tables'].items():
            out.__dict__['tables'][k] = v
        return out

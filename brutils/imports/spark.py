import pyspark.sql.functions as F
import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import Window


class E:
    def __getattr__(self, x):
        out = F.expr2(x)
        setattr(self, x, out)
        return out

    def __call__(self, x):
        return F.expr2(x)


class M:
    def __getattr__(self, x):
        out = F.col(x)
        setattr(self, x, out)
        return out
m = M()
e = E()

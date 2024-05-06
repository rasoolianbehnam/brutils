import re
import rpy2
import numpy as np
import pandas as pd

def r2Py(x):
    tp = str(type(x)).split(".")[-1][:-2]
    if re.match(".*ListVector$", tp):
        return pd.Series({k: r2Py(v) for k, v in x.items()})
    if re.match(".*(Array|Vector|Matrix)$", tp):
        return np.asarray(x.memoryview())
    if re.match("NULLType$", tp):
        return None
    return x


rpy2.robjects.vectors.ListVector.v = property(r2Py)

import random
import joblib
import operator
import functools
import collections
import itertools
import os, sys
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import seaborn as sns
import seaborn as sn
import matplotlib.pyplot as plt
from functools import reduce
from functools import partial
from datetime import datetime
from datetime import timedelta
import time
import scipy
from pathlib import Path
import datetime as dt


class S_:
    def __getattr__(self, x):
        setattr(self, x, x)
        return x


s = S_()
c = np.array


def alarm(t=5):
    time.sleep(t)


import ray
from types import ModuleType
import importlib
from functools import wraps, partial

import os, subprocess


class remote_ray_jupyter:
    def __init__(self, n_returns=1, *args, **kwargs):
        self.n_returns = n_returns

    def __call__(self, fun):
        @wraps(fun)
        def g(*args, **kwargs):
            out = f"/tmp/ray/session_latest/logs/worker*{os.getpid()}.out"
            err = f"/tmp/ray/session_latest/logs/worker*{os.getpid()}.err"
            nout = int(subprocess.getoutput(f"cat {out} | wc -m").split()[-1])
            nerr = int(subprocess.getoutput(f"cat {err} | wc -m").split()[-1])
            for i in range(2):
                for k, v in fun.__globals__.items():
                    if isinstance(v, ModuleType):
                        importlib.reload(v)
            ret = fun(*args, **kwargs)
            if self.n_returns == 1:
                ret = [ret]
            return (
                subprocess.getoutput(f"cat {out}")[nout:],
                subprocess.getoutput(f"cat {err}")[nerr:],
                *ret,
            )

        return ray.remote(num_returns=2 + self.n_returns)(g)


class Scheduler:
    def __init__(self, n):
        self.n = n
        self.running = []

    @wraps(ray.remote)
    def __call__(self, *args, **kwargs):
        if len(args):
            if len(kwargs):
                f = ray.remote(**kwargs)(*args)
            else:
                f = ray.remote(*args)
        else:
            return partial(self, **kwargs)

        @wraps(f.remote)
        def foo(*args, **kwargs):
            return self.submit(f, *args, **kwargs)

        return foo

    def submit(self, fun, *args, **kwargs):
        k = len(self.running) - self.n + 1
        self.wait(k)
        res = out = fun.remote(*args, **kwargs)
        if isinstance(res, list):
            res = out[0]
        self.running.append(res)
        return out

    def wait(self, k=None):
        if k is None:
            _, self.running = ray.wait(self.running, num_returns=len(self.running))
        elif k >= 1:
            _, self.running = ray.wait(self.running, num_returns=k)

    def reset(self, n):
        self.n = n
        k = len(self.running) - self.n + 1
        self.wait(k)

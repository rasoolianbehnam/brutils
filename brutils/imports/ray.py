import ray
from types import ModuleType
import importlib
from functools import wraps
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
    def __call__(self, fun, *args, **kwargs):
        f = ray.remote(fun, *args, **kwargs)

        @wraps(fun)
        def foo(*args, **kwargs):
            return self.submit(f, *args, **kwargs)

        return foo

    def submit(self, fun, *args, **kwargs):
        while len(self.running) >= self.n:
            _, self.running = ray.wait(self.running, num_returns=1)
        res = fun.remote(*args, **kwargs)
        self.running.append(res)
        return res

    def reset(self):
        self.running = []

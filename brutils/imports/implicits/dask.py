from brutils.utility import RegisterWithClass
import dask.dataframe as dd


@RegisterWithClass(dd.DataFrame)
def Rename2(self, *args, **kwargs):
    if len(args):
        return self.rename(columns={v: k for k, v in args[0].items()})
    return self.rename(columns={v: k for k, v in kwargs.items()})


@RegisterWithClass(dd.DataFrame)
def dropcol(self, *args, **kwargs):
    if isinstance(args[0], list):
        args = args[0]
    return self.drop(list(args), axis=1, **kwargs)


dd.DataFrame.pandas = dd.DataFrame.compute

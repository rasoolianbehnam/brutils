try:
    # noinspection PyUnresolvedReferences
    from IPython import get_ipython
    if get_ipython() is not None:
        try:
            get_ipython().run_line_magic('load_ext', 'julia.magic')
        except Exception:
            from julia.api import Julia
            jl = Julia(compiled_modules=False)
            get_ipython().run_line_magic('load_ext', 'julia.magic')
        from julia import Main
        import site
        import os
        os.environ["JULIA_NUM_THREADS"] = "-1"
        os.environ["OPENBLAS_NUM_THREADS"] = "-1"
        get_ipython().run_line_magic('config', 'JuliaMagics.revise=True')
        res = get_ipython().getoutput('cat {site.getsitepackages()[0]}/weight-calculation-lp*')
        julia_package_path = f"{res[0]}/brutils/resources/WeightCalculationLp/"
        get_ipython().run_line_magic('julia', 'using Pkg; Pkg.activate($julia_package_path)')

except ImportError:
    pass



import glob
import os
import re
import time
import argparse

import psutil

try:
    # noinspection PyUnresolvedReferences
    from IPython import get_ipython
    from IPython.core.magic import (register_line_magic, register_cell_magic,
                                    register_line_cell_magic, needs_local_scope)

    if get_ipython() is not None:
        print("Ipython!")
        get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')


    @register_cell_magic
    def jj(line, cell):
        "my cell magic"
        get_ipython().run_line_magic('julia', "revise()")
        return get_ipython().run_cell_magic('julia', line, cell)


    @register_line_magic
    def jj(line):
        "my cell magic"
        get_ipython().run_line_magic('julia', "revise()")
        return get_ipython().run_line_magic('julia', line)
except ImportError:
    pass


def enable_julia(julia_path=None, julia_depot_path="", n_threads=""):
    if julia_path:
        if julia_path not in os.environ['PATH']:
            os.environ['PATH'] += f":{julia_path}"
    if julia_depot_path:
        os.environ["JULIA_DEPOT_PATH"] = julia_depot_path
    os.environ["JULIA_NUM_THREADS"] = n_threads or str(psutil.cpu_count() // 2)
    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count() // 2)
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    # get_ipython().run_line_magic('config', 'JuliaMagics.revise = True')
    # get_ipython().run_line_magic('load_ext', 'julia.magic')
    # get_ipython().run_line_magic('julia', 'using Revise')
    # d = os.path.dirname(os.path.realpath(__file__))
    # files = glob.glob(f'{d}/../resources/WeightCalculationLp/src/*jl')
    # for file in files:
    #     Main.eval(f'includet("{file}")')
    return Main


def enable_magic_continuation():
    get_ipython().run_cell_magic('javascript', '',
                                 '   \nvar d = {\n    help : \'run cell\',\n    help_index : \'zz\',\n    handler : function (event) {\n        var line=IPython.notebook.get_selected_cell().code_mirror.doc.getLine(0);\n        console.log(line.startsWith("%%"))\n        IPython.notebook.execute_cell_and_select_below();\n        if (line.startsWith("%%")) {\n            var line_content=IPython.notebook.get_selected_cell().code_mirror.doc.getLine(0);\n            var cm=IPython.notebook.get_selected_cell().code_mirror;\n            var line_content = cm.doc.getLine(0);\n            console.log(line_content)\n            if (!line_content) {\n                cm.doc.replaceSelection(line);\n                cm.execCommand(\'newlineAndIndent\');\n            }\n            var current_cursor = cm.doc.getCursor();\n            cm.doc.setCursor(current_cursor.line + 1, current_cursor.ch);\n        }\n        return false;\n    }}\nJupyter.keyboard_manager.command_shortcuts.add_shortcut(\'shift-enter\', d)\nJupyter.keyboard_manager.edit_shortcuts.add_shortcut(\'shift-enter\', d)')


def get_notebook_name():
    get_ipython().run_cell_magic('javascript', '',
                                 """IPython.notebook.kernel.execute('nb_name = "' + IPython.notebook.notebook_name + '"')""")
    get_ipython().run_cell_magic('javascript', '',
                                 """IPython.notebook.kernel.execute('nb_name = nb_name.replace(".ipynb", "")')""")


@register_cell_magic
@needs_local_scope  # to access local variables
def model(line, cell, local_ns):
    """
    for new model add args=()
    -v for verbose
    """
    parser = argparse.ArgumentParser(prefix_chars="/")
    parser.add_argument('/args',
                        default='',
                        dest='model_args',
                        help='Provide arguments that go into pm.Model',
                        type=str
                        )
    parser.add_argument('/v', "//verbose",
                        dest="verbose",
                        action="store_true"
                        )
    parser.add_argument('/n', "//new",
                        dest="new_model",
                        action="store_true"
                        )
    parser.add_argument('/d', "//data",
                        default='',
                        dest='data',
                        help='Data',
                        type=str
                        )
    parser.add_argument('model_name')
    args = parser.parse_args(line.split())
    locals().update(local_ns)

    if not cell.strip(): cell = "pass"
    cell = ''.join('\n\t' + x for x in cell.strip().split('\n'))

    new_model = args.model_name not in local_ns or args.new_model
    model_args = args.model_args
    if args.data:
        if 'coords=' not in model_args:
            model_args += f"coords={args.data}.coords(),"
    model_line = f"pm.Model({model_args}) as {args.model_name}" if new_model else args.model_name
    cmd = f"with {model_line}:\n\tmodel_ = {args.model_name}\n"
    if args.data:
        cmd += f"\tdata = {args.data}.Data()\n"
    cmd += cell
    if args.verbose:
        print(cmd)

    exec(cmd)
    l = {k: v for k, v in locals().items()}
    del l["local_ns"]
    del l["cell"]
    del l["line"]
    local_ns.update(l)


def remove_warnings():
    get_ipython().run_cell_magic('javascript', '',
                                 '(function(on) {\nconst e=$( "<a>Setup failed</a>" );\nconst ns="js_jupyter_suppress_warnings";\nvar cssrules=$("#"+ns);\nif(!cssrules.length) cssrules = $("<style id=\'"+ns+"\' type=\'text/css\'>div.output_stderr { } </style>").appendTo("head");\ne.click(function() {\n    var s=\'Showing\';  \n    cssrules.empty()\n    if(on) {\n        s=\'Hiding\';\n        cssrules.append("div.output_stderr, div[data-mime-type*=\'.stderr\'] { display:none; }");\n    }\n    e.text(s+\' warnings (click to toggle)\');\n    on=!on;\n}).click();\n$(element).append(e);\n})(true);')


@register_line_magic
@needs_local_scope
def thread(line, local_ns):
    import brutils.utility as ut
    print(local_ns.keys())
    locals().update(local_ns)
    lines = line.split(";")
    command = f"ut.run_in_thread2(lambda: {lines[0]})"
    a = eval(command)
    b = eval(lines[1])
    return a.get(), b


@register_cell_magic
@needs_local_scope
def dictionary(args, cell, local_ns):
    if not args:
        raise ValueError("should provide an output name.")
    out = {}
    for line in cell.strip().split("\n"):
        l, r = line.split("=")
        l, r = l.strip(), r.strip()
        out[l] = eval(r)
    return out

# enable_magic_continuation()
get_notebook_name()

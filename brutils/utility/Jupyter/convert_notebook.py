#!/usr/bin/env python
# coding: utf-8


import os
import argparse
import json
import copy


def convert(args=None):
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--input-file", type=str, help="Input File.")
    parser.add_argument("--output-file", type=str, help="Output File.")

    args = parser.parse_args(args)

    with open(args.input_file) as f:
        data_original = json.load(f)

    data = copy.deepcopy(data_original)

    cells = data["cells"]

    def isMarkdown(cell):
        return cell["cell_type"] == "markdown"

    def isCode(cell):
        return cell["cell_type"] == "code"

    def startswith(cell, s):
        data = cell["source"]
        if len(data):
            return data[0].strip().lower().startswith(s.lower())
        return False

    def hasCellMagic(
        cell,
    ):
        return isCode(cell) and startswith(cell, "%%")

    def frozen(cell):
        return cell["metadata"].get("frozen", False)

    starts = [
        i
        for i, cell in enumerate(cells)
        if isMarkdown(cell) and startswith(cell, "#### remove")
    ]
    ends = [
        i
        for i, cell in enumerate(cells)
        if isMarkdown(cell) and startswith(cell, "#### end remove")
    ]
    assert len(starts) == len(ends)

    to_remove = set()
    for i, j in zip(starts, ends):
        to_remove.add(i)
        to_remove.add(j)
        for k in range(i + 1, j):
            if not frozen(cells[k]):
                to_remove.add(k)

    start = [
        i
        for i, cell in enumerate(cells)
        if isMarkdown(cell) and startswith(cell, "# draft")
    ][0]
    for i in range(start, len(cells)):
        to_remove.add(i)

    for cell in cells:
        if hasCellMagic(cell):
            cell["source"] = cell["source"][1:]

    new_cells = [c for i, c in enumerate(cells) if i not in to_remove]

    cells[0]

    functions = [
        [cell["source"][0][2:].strip().lower().replace(" ", "_"), i + 1, 0]
        for i, cell in enumerate(new_cells)
        if isMarkdown(cell) and startswith(cell, "# ")
    ]

    for i in range(1, len(functions)):
        functions[i - 1][2] = functions[i][1]

    functions[-1][2] = len(new_cells)

    functions

    new_cells[4]["source"]

    def join_cells(cells, indent=""):
        out = []
        for cell in cells:
            source = "".join([f"{indent}{line}" for line in cell["source"]])
            out.append("".join(source))
        return "\n\n".join(out)

    body = [join_cells(new_cells[: functions[0][1]])]
    for name, s, f in functions:
        out = join_cells([new_cells[i] for i in range(s, f)], indent="\t")
        out = f"def {name}():\n{out}"
        body.append(out)

    in_main = "\n\t".join([f"{f[0]}()" for f in functions])

    final_body = "\n\n".join(body)
    final_body = f"""#!/usr/bin/env python
# coding: utf-8

{final_body}

if __name__=='__main__':
\t{in_main}
"""

    # print(final_body)

    with open(args.output_file, "w") as f:
        f.write(final_body)

    os.system(f"black {args.output_file}")

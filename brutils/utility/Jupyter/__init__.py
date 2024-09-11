import json
import pandas as pd


def get_total_execution_time(path_to_ipynb, header=None, unit='s'):
    denuminator = {'s': 1, 'h': 3600, 'm': 60}.get(unit, 1)
    with open(path_to_ipynb) as f:
        data = json.load(f)
    if header is None:
        return round(sum(_extract_durations(data['cells'])) / denuminator, 2)
    d = {'': []}
    out = d['']
    for cell in data['cells']:
        if cell['cell_type'] == 'markdown' and cell['source'][0].startswith(header):
            out = d[cell['source'][0]] = []
        else:
            out.append(_extract_duration(cell))
    return {k: round(sum(v)/denuminator, 2) for k, v in d.items()}


def _extract_durations(cells):
    for cell in cells:
        yield _extract_duration(cell)


def _extract_duration(cell):
    try:
        execution = cell['metadata']['execution']
        return (pd.Timestamp(execution['iopub.status.idle']) - pd.Timestamp(
                execution['iopub.execute_input'])).total_seconds()
    except:
        return 0

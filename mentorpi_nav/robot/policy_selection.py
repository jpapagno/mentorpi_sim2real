"""Resolve a named installed policy without changing the active model files."""
import json
from pathlib import Path


def resolve_policy(root,name='active'):
    models=(Path(root)/'models').resolve()
    if name=='active':directory=models
    else:
        catalog=json.loads((models/'catalog.json').read_text())
        relative=catalog['policies'].get(name)
        if not isinstance(relative,str):raise ValueError(f'Unknown installed policy: {name}')
        directory=(models/relative).resolve()
        if directory==models or models not in directory.parents:raise ValueError('Policy directory escapes models')
    for filename in ('policy.ts','contract.json'):
        if not (directory/filename).is_file():raise ValueError(f'Missing policy file: {directory/filename}')
    return directory

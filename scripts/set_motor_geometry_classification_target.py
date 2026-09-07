#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mor.nn.data.motorFullResolutionDataset import _geometryClassificationTask


TASK_NAME = 'geometryComponentClassification'


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'dataset index must contain a JSON object: {path}')
    return value


def _updatedIndex(path: Path) -> tuple[dict, bool]:
    index = _load(path)
    entries = index.get('cases')
    if not isinstance(entries, list) or not entries:
        raise ValueError(f'dataset index has no component cases: {path}')
    task = _geometryClassificationTask(entries)
    if not task['classes']:
        raise ValueError(f'dataset index has no componentType labels: {path}')
    tasks = index.get('tasks', {})
    if not isinstance(tasks, dict):
        raise ValueError(f'dataset tasks must contain a JSON object: {path}')
    changed = tasks.get(TASK_NAME) != task
    index['tasks'] = {**tasks, TASK_NAME: task}
    return index, changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Declare componentType as the geometry classification target.'
    )
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    arguments = parser.parse_args()

    results = []
    for view in ('point', 'graph'):
        path = arguments.dataset / view / 'dataset.json'
        if not path.is_file():
            raise FileNotFoundError(f'dataset index is missing: {path}')
        index, changed = _updatedIndex(path)
        task = index['tasks'][TASK_NAME]
        if changed and not arguments.dry_run:
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(index, indent=2), encoding='utf-8')
            temporary.replace(path)
        results.append(
            {
                'view': view,
                'path': str(path.resolve()),
                'changed': changed,
                'written': changed and not arguments.dry_run,
                'targetField': task['targetField'],
                'classes': task['classes'],
            }
        )
    print(json.dumps({'dataset': str(arguments.dataset.resolve()), 'views': results}, indent=2))


if __name__ == '__main__':
    main()

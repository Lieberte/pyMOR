from __future__ import annotations

import argparse
import json
from pathlib import Path

from mor.nn.data import buildFullResolutionMotorDatasetFromDirectories


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build full-resolution point and graph motor datasets without subsampling.'
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--motor-id', required=True)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--components', nargs='*')
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    viewsPath = buildFullResolutionMotorDatasetFromDirectories(
        arguments.source,
        arguments.output,
        motorId=arguments.motor_id,
        caseIds=arguments.cases,
        components=arguments.components,
    )
    views = json.loads(viewsPath.read_text(encoding='utf-8'))
    print(json.dumps({'views': str(viewsPath), **views}, indent=2))


if __name__ == '__main__':
    main()

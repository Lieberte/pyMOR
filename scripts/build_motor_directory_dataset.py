from __future__ import annotations

import argparse
import json
from pathlib import Path

from mor.nn.data import buildMotorDatasetFromDirectories, loadThermalComponentDataset


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert numeric solved-motor directories into component and whole-motor views.'
    )
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--motor-id', required=True)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--components', nargs='*')
    parser.add_argument('--maximum-volume-samples', type=int)
    parser.add_argument('--maximum-boundary-samples', type=int)
    parser.add_argument('--maximum-query-samples', type=int)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    build = buildMotorDatasetFromDirectories(
        arguments.source,
        arguments.output,
        motorId=arguments.motor_id,
        caseIds=arguments.cases,
        components=arguments.components,
        maximumVolumeSamples=arguments.maximum_volume_samples,
        maximumBoundarySamples=arguments.maximum_boundary_samples,
        maximumQuerySamples=arguments.maximum_query_samples,
        randomSeed=arguments.random_seed,
        overwrite=arguments.overwrite,
    )
    cases = loadThermalComponentDataset(arguments.output)
    print(
        json.dumps(
            {
                'componentDataset': str(build.componentIndexPath),
                'wholeMotorDataset': str(build.wholeMotorIndexPath),
                'views': str(build.viewsPath),
                'motorId': build.motorId,
                'parentCaseCount': len(build.parentCaseIds),
                'parentCaseIds': build.parentCaseIds,
                'componentCaseCount': len(cases),
                'components': sorted({case.componentType for case in cases}),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()

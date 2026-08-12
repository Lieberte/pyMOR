from __future__ import annotations

import argparse
import json
from pathlib import Path

from mor.nn.data import (
    CONVECTION_BOUNDARY_TYPE,
    MOTOR_FIXED_TEMPERATURE_BOUNDARY_TYPE,
    RESULT_HEAT_FLUX_BOUNDARY_TYPE,
    buildMotorFemComponentCases,
    formatMotorFemBoundaryHeatFlowReport,
    saveThermalComponentDataset,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert solved motor FEM regions into independent thermal component cases.'
    )
    parser.add_argument('source', type=Path, help='directory containing numeric speed cases')
    parser.add_argument('output', type=Path, help='thermal component dataset output directory')
    parser.add_argument('--component', required=True, help='region name, short name, or UUID')
    parser.add_argument(
        '--speeds', nargs='*', help='speed directory names; default: all numeric cases'
    )
    parser.add_argument('--maximum-volume-samples', type=int)
    parser.add_argument('--maximum-boundary-samples', type=int)
    parser.add_argument('--maximum-query-samples', type=int)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='print per-surface heat-flow and area-integration checks',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    cases = buildMotorFemComponentCases(
        arguments.source,
        arguments.component,
        speeds=arguments.speeds,
        maximumVolumeSamples=arguments.maximum_volume_samples,
        maximumBoundarySamples=arguments.maximum_boundary_samples,
        maximumQuerySamples=arguments.maximum_query_samples,
        randomSeed=arguments.random_seed,
    )
    indexPath = saveThermalComponentDataset(
        cases,
        arguments.output,
        boundaryTypeNames={
            RESULT_HEAT_FLUX_BOUNDARY_TYPE: 'resultHeatFlux',
            CONVECTION_BOUNDARY_TYPE: 'convection',
            MOTOR_FIXED_TEMPERATURE_BOUNDARY_TYPE: 'fixedTemperature',
        },
        overwrite=arguments.overwrite,
    )
    summary = {
        'dataset': str(indexPath),
        'caseCount': len(cases),
        'cases': [
            {
                'caseId': case.caseId,
                'geometryId': case.geometryId,
                'volumeSamples': int(case.volume.coordinatesMeters.shape[0]),
                'boundarySamples': int(case.boundary.coordinatesMeters.shape[0]),
                'querySamples': int(case.temperature.coordinatesMeters.shape[0]),
                'configuredHeatSourcePowerW': case.metadata['configuredHeatSourcePowerW'],
                'integratedHeatSourcePowerW': case.volume.totalHeatSourcePowerW,
                'exportedBoundaryHeatFlowRateW': case.metadata[
                    'exportedBoundaryHeatFlowRateW'
                ],
                'integratedBoundaryHeatFlowRateW': case.boundary.totalHeatFlowRateW,
                'boundaryHeatFlowErrorW': case.metadata['boundaryHeatFlowErrorW'],
                'surfacePatchCount': case.metadata['surfacePatchCount'],
            }
            for case in cases
        ],
    }
    print(json.dumps(summary, indent=2))
    if arguments.verbose:
        for case in cases:
            print()
            print(formatMotorFemBoundaryHeatFlowReport(case))


if __name__ == '__main__':
    main()

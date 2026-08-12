from __future__ import annotations

import argparse
import json
from pathlib import Path

from mor.nn.data import (
    buildMotorDatasetFromSta,
    formatMotorFemBoundaryHeatFlowReport,
    loadThermalComponentDataset,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert one STA motor result into independent FEM component cases.'
    )
    parser.add_argument('source', type=Path, help='source .sta archive')
    parser.add_argument('output', type=Path, help='single-motor dataset output directory')
    parser.add_argument('--motor-id')
    parser.add_argument('--case-id')
    parser.add_argument('--components', nargs='*')
    parser.add_argument('--result-time', type=float)
    parser.add_argument('--result-member')
    parser.add_argument(
        '--extracted-directory',
        type=Path,
        help='persistent extraction/cache directory; defaults to a complete sibling directory or a temporary directory',
    )
    parser.add_argument('--maximum-volume-samples', type=int)
    parser.add_argument('--maximum-boundary-samples', type=int)
    parser.add_argument('--maximum-query-samples', type=int)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    build = buildMotorDatasetFromSta(
        arguments.source,
        arguments.output,
        motorId=arguments.motor_id,
        parentCaseId=arguments.case_id,
        components=arguments.components,
        resultTime=arguments.result_time,
        resultMember=arguments.result_member,
        extractedDirectory=arguments.extracted_directory,
        maximumVolumeSamples=arguments.maximum_volume_samples,
        maximumBoundarySamples=arguments.maximum_boundary_samples,
        maximumQuerySamples=arguments.maximum_query_samples,
        randomSeed=arguments.random_seed,
        overwrite=arguments.overwrite,
    )
    cases = loadThermalComponentDataset(arguments.output)
    summary = {
        'componentDataset': str(build.indexPath),
        'wholeMotorDataset': str(build.wholeMotorIndexPath),
        'views': str(build.viewsPath),
        'motorId': build.motorId,
        'parentCaseId': build.parentCaseId,
        'selectedResultMember': build.resultMember,
        'reusedExtractedDirectory': (
            str(build.reusedExtractedDirectory)
            if build.reusedExtractedDirectory is not None
            else None
        ),
        'componentCount': len(cases),
        'components': [
            {
                'caseId': case.caseId,
                'componentType': case.componentType,
                'volumeSamples': int(case.volume.coordinatesMeters.shape[0]),
                'boundarySamples': int(case.boundary.coordinatesMeters.shape[0]),
                'querySamples': int(case.temperature.coordinatesMeters.shape[0]),
                'absoluteTemperatureAnchorAvailable': case.metadata[
                    'absoluteTemperatureAnchorAvailable'
                ],
                'integratedHeatSourcePowerW': case.volume.totalHeatSourcePowerW,
                'integratedBoundaryHeatFlowRateW': case.boundary.totalHeatFlowRateW,
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

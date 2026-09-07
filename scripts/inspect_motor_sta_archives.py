from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from mor.nn.data import inspectMotorStaArchive


def schemaHints(path: Path) -> dict:
    with ZipFile(path) as archive:
        control = json.loads(archive.read('controlData.json').decode('utf-8-sig'))
        root = control.get('rootSystem', {})
        thermal = root.get('thermalComponents', {})
        directRegions = thermal.get('femRegions', {})
        matchingPaths = []
        stack = [('', control)]
        while stack and len(matchingPaths) < 80:
            prefix, value = stack.pop()
            if not isinstance(value, dict):
                continue
            for key, child in value.items():
                pathName = f'{prefix}/{key}' if prefix else str(key)
                if 'fem' in str(key).lower() or 'thermal' in str(key).lower():
                    matchingPaths.append(pathName)
                if isinstance(child, dict):
                    stack.append((pathName, child))
        archiveNames = [info.filename for info in archive.infolist() if not info.is_dir()]
        return {
            'rootSystemKeys': sorted(root),
            'thermalComponentKeys': sorted(thermal),
            'directFemRegionCount': len(directRegions),
            'directFemRegions': [
                {
                    'uuid': str(uuid),
                    'name': region.get('name'),
                    'meshUuid': region.get('meshUuid'),
                    'materialUuid': region.get('materialUuid'),
                    'keys': sorted(region),
                }
                for uuid, region in directRegions.items()
            ],
            'matchingControlPaths': sorted(matchingPaths),
            'meshFileCount': sum(name.startswith('mesh/') for name in archiveNames),
            'resultFileCount': sum(name.startswith('results/') and name.endswith('.h5') for name in archiveNames),
            'sampleMeshMembers': [name for name in archiveNames if name.startswith('mesh/')][:12],
            'sampleResultMembers': [name for name in archiveNames if name.startswith('results/') and name.endswith('.h5')][:12],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect motor STA archives without extracting them.')
    parser.add_argument('directory', type=Path)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--concise', action='store_true')
    arguments = parser.parse_args()

    pattern = '**/*.sta' if arguments.recursive else '*.sta'
    records = []
    for path in sorted(arguments.directory.glob(pattern)):
        try:
            info = inspectMotorStaArchive(path)
            records.append({
                'file': path.name,
                'path': str(path),
                'sizeBytes': path.stat().st_size,
                'resultMember': info.resultMember,
                'resultTime': info.resultTime,
                'analysisType': info.analysisType,
                'isTransient': info.isTransient,
                'sourceTimeStepS': info.sourceTimeStepS,
                'sourceEndTimeS': info.sourceEndTimeS,
                'availableResultCount': len(info.availableResultCoordinates),
                'components': info.componentNames,
                'meshCount': len(info.meshMembers),
            })
        except Exception as error:
            hints = schemaHints(path)
            records.append({
                'file': path.name,
                'path': str(path),
                'error': f'{type(error).__name__}: {error}',
                'schema': (
                    {
                        'directFemRegionCount': hints['directFemRegionCount'],
                        'directFemRegions': [
                            {
                                'uuid': region['uuid'],
                                'name': region['name'],
                                'meshUuid': region['meshUuid'],
                            }
                            for region in hints['directFemRegions']
                        ],
                        'meshFileCount': hints['meshFileCount'],
                        'resultFileCount': hints['resultFileCount'],
                        'sampleResultMembers': hints['sampleResultMembers'],
                    }
                    if arguments.concise
                    else hints
                ),
            })
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()

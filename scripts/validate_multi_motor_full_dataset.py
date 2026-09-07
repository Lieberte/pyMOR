from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def validate(root: Path) -> dict:
    root = root.resolve()
    views = _load(root / 'views.json')
    hierarchyPath = root / views.get('hierarchyCatalog', 'hierarchy.json')
    hierarchy = _load(hierarchyPath) if hierarchyPath.is_file() else {}
    motorIds = set(views.get('motorIds', ()))
    parentCaseIds = set(views.get('parentCaseIds', ()))
    errors = []
    componentKeysByView = {}
    parentAnalysisByView = {}
    typeCounts = Counter()
    npzCount = 0
    npzBytes = 0
    seriesCount = 0
    seriesBytes = 0

    if not views.get('fullResolution') or not views.get('selfContainedCases'):
        errors.append('root views must declare full-resolution self-contained cases')
    if views.get('geometryAssetsSharedByMeshUuid') is not False:
        errors.append('root views must declare that geometry assets are not shared')

    for view in ('point', 'graph'):
        indexPath = root / view / 'dataset.json'
        index = _load(indexPath)
        entries = list(index.get('cases', ()))
        keys = {(entry.get('motorId'), entry.get('caseId')) for entry in entries}
        if len(keys) != len(entries):
            errors.append(f'{view} component index contains duplicate motor/case identities')
        componentKeysByView[view] = keys
        byParent = {}
        analysisByParent = {}
        for entry in entries:
            motorId = entry.get('motorId')
            parentId = entry.get('parentCaseId')
            caseId = entry.get('caseId')
            if motorId not in motorIds or parentId not in parentCaseIds:
                errors.append(f'{view} has unknown motor/parent identity: {motorId}/{parentId}')
            byParent.setdefault((motorId, parentId), set()).add(caseId)
            analysisType = entry.get('analysisType')
            if analysisType not in {'steady', 'transient'}:
                errors.append(f'{view} has invalid analysisType: {caseId}/{analysisType}')
            parentKey = (motorId, parentId)
            previousAnalysis = analysisByParent.setdefault(parentKey, analysisType)
            if previousAnalysis != analysisType:
                errors.append(f'{view} has mixed analysis types in parent case: {parentKey}')
            caseDirectory = indexPath.parent / entry.get('path', '')
            if not caseDirectory.is_dir():
                errors.append(f'missing {view} component directory: {caseDirectory}')
                continue
            manifestPath = caseDirectory / 'manifest.json'
            dataPath = caseDirectory / 'data.npz'
            if not manifestPath.is_file() or not dataPath.is_file():
                errors.append(f'incomplete {view} component directory: {caseDirectory}')
                continue
            manifest = _load(manifestPath)
            for field, expected in (
                ('caseId', caseId),
                ('motorId', motorId),
                ('parentCaseId', parentId),
                ('componentType', entry.get('componentType')),
                ('componentInstance', entry.get('componentInstance')),
                ('analysisType', analysisType),
                ('temporalSampleType', entry.get('temporalSampleType')),
                ('systemType', entry.get('systemType')),
                ('assemblyType', entry.get('assemblyType')),
                ('assemblyId', entry.get('assemblyId')),
            ):
                if manifest.get(field) != expected:
                    errors.append(f'{view} manifest {field} mismatch: {caseDirectory}')
            if not manifest.get('componentType') or not manifest.get('componentInstance'):
                errors.append(f'{view} manifest lacks component labels: {caseDirectory}')
            if manifest.get('metadata', {}).get('analysisType') != analysisType:
                errors.append(f'{view} manifest metadata analysisType mismatch: {caseDirectory}')
            if analysisType == 'transient':
                if entry.get('timeSeriesStored'):
                    seriesPath = caseDirectory / entry.get('seriesFile', 'series.h5')
                    if not seriesPath.is_file():
                        errors.append(f'{view} transient series is missing: {seriesPath}')
                    else:
                        with h5py.File(seriesPath, 'r') as series:
                            requiredSeries = {
                                'timeS', 'nodeTemperatureK', 'boundaryHeatFluxWPerM2'
                            }
                            missingSeries = requiredSeries - set(series)
                            if missingSeries:
                                errors.append(
                                    f'{view} transient series lacks {sorted(missingSeries)}: {seriesPath}'
                                )
                            else:
                                times = np.asarray(series['timeS'], dtype=np.float64)
                                expectedCount = int(entry.get('timeStepCount', -1))
                                if (
                                    times.size != expectedCount
                                    or times.size < 2
                                    or np.any(np.diff(times) <= 0.0)
                                ):
                                    errors.append(f'{view} transient times are invalid: {seriesPath}')
                                if series['nodeTemperatureK'].shape[0] != times.size:
                                    errors.append(f'{view} transient temperature length mismatch: {seriesPath}')
                                if series['boundaryHeatFluxWPerM2'].shape[0] != times.size:
                                    errors.append(f'{view} transient boundary length mismatch: {seriesPath}')
                                if str(series.attrs.get('meshUuid', '')) != str(manifest.get('meshUuid', '')):
                                    errors.append(f'{view} transient meshUuid mismatch: {seriesPath}')
                                for hierarchyField in ('systemType', 'assemblyType', 'assemblyId'):
                                    if str(series.attrs.get(hierarchyField, '')) != str(entry.get(hierarchyField, '')):
                                        errors.append(
                                            f'{view} transient {hierarchyField} mismatch: {seriesPath}'
                                        )
                                with np.load(dataPath, allow_pickle=False) as finalData:
                                    if not np.allclose(
                                        series['nodeTemperatureK'][-1],
                                        finalData['nodeTemperatureK'],
                                        rtol=0.0,
                                        atol=0.0,
                                    ):
                                        errors.append(f'{view} transient terminal temperature mismatch: {seriesPath}')
                                    if not np.allclose(
                                        series['boundaryHeatFluxWPerM2'][-1],
                                        finalData['boundaryHeatFluxWPerM2'],
                                        rtol=5e-11,
                                        atol=1e-9,
                                    ):
                                        errors.append(f'{view} transient terminal boundary mismatch: {seriesPath}')
                        seriesCount += 1
                        seriesBytes += seriesPath.stat().st_size
                    if 'snapshotTimeS' in manifest.get('operatingFeatures', {}):
                        errors.append(
                            f'{view} transient sequence unexpectedly has a case-level snapshot time: {caseDirectory}'
                        )
                else:
                    snapshotTime = entry.get('snapshotTimeS')
                    if snapshotTime is None or manifest.get('operatingFeatures', {}).get('snapshotTimeS') != snapshotTime:
                        errors.append(f'{view} transient snapshot time is missing or inconsistent: {caseDirectory}')
            elif 'snapshotTimeS' in manifest.get('operatingFeatures', {}):
                errors.append(f'{view} steady case unexpectedly has snapshotTimeS: {caseDirectory}')
            if view == 'point':
                typeCounts[manifest.get('componentType')] += 1
            try:
                with zipfile.ZipFile(dataPath) as archive:
                    names = set(archive.namelist())
                    required = (
                        {'queryCoordinatesMeters.npy', 'nodeTemperatureK.npy'}
                        if view == 'point'
                        else {'nodesMeters.npy', 'tetrahedra.npy', 'surfaceTriangles.npy'}
                    )
                    missing = required - names
                    if missing:
                        errors.append(f'{view} NPZ lacks {sorted(missing)}: {dataPath}')
            except zipfile.BadZipFile:
                errors.append(f'invalid NPZ container: {dataPath}')
            npzCount += 1
            npzBytes += dataPath.stat().st_size

        parentAnalysisByView[view] = analysisByParent

        wholeIndexPath = root / view / 'whole_motor' / 'dataset.json'
        wholeIndex = _load(wholeIndexPath)
        wholeEntries = list(wholeIndex.get('cases', ()))
        wholeKeys = {(entry.get('motorId'), entry.get('parentCaseId')) for entry in wholeEntries}
        if len(wholeKeys) != len(wholeEntries):
            errors.append(f'{view} whole-motor index contains duplicates')
        if wholeKeys != set(byParent):
            errors.append(f'{view} whole-motor cases do not match component parent groups')
        for entry in wholeEntries:
            motorId = entry.get('motorId')
            parentId = entry.get('parentCaseId')
            caseDirectory = wholeIndexPath.parent / entry.get('path', '')
            manifestPath = caseDirectory / 'manifest.json'
            if not manifestPath.is_file():
                errors.append(f'missing {view} whole-motor manifest: {manifestPath}')
                continue
            manifest = _load(manifestPath)
            if manifest.get('motorId') != motorId or manifest.get('parentCaseId') != parentId:
                errors.append(f'{view} whole-motor identity mismatch: {caseDirectory}')
            for hierarchyField in ('systemType', 'assemblyType', 'assemblyId'):
                if manifest.get(hierarchyField) != entry.get(hierarchyField):
                    errors.append(
                        f'{view} whole-assembly {hierarchyField} mismatch: {caseDirectory}'
                    )
            expectedAnalysis = analysisByParent.get((motorId, parentId))
            if entry.get('analysisType') != expectedAnalysis or manifest.get('analysisType') != expectedAnalysis:
                errors.append(f'{view} whole-motor analysisType mismatch: {caseDirectory}')
            expectedComponents = byParent.get((motorId, parentId), set())
            actualComponents = {component.get('caseId') for component in manifest.get('components', ())}
            if actualComponents != expectedComponents:
                errors.append(f'{view} whole-motor component set mismatch: {caseDirectory}')
            for component in manifest.get('components', ()):
                componentPath = (caseDirectory / component.get('path', '')).resolve()
                if not componentPath.is_dir():
                    errors.append(f'broken {view} whole-motor component path: {componentPath}')
            for interface in manifest.get('interfaces', ()):
                for side in interface.get('sides', ()):
                    for field in ('componentCaseId', 'oppositeComponentCaseId'):
                        value = side.get(field)
                        if value is not None and value not in expectedComponents:
                            errors.append(f'broken {view} interface {field}: {value}')

        for analysisType in ('steady', 'transient'):
            analysisIndexPath = root / view / analysisType / 'dataset.json'
            analysisWholePath = root / view / analysisType / 'whole_motor' / 'dataset.json'
            if not analysisIndexPath.is_file() or not analysisWholePath.is_file():
                errors.append(f'missing {view}/{analysisType} analysis indexes')
                continue
            analysisIndex = _load(analysisIndexPath)
            analysisEntries = list(analysisIndex.get('cases', ()))
            expectedAnalysisKeys = {
                (entry.get('motorId'), entry.get('caseId'))
                for entry in entries
                if entry.get('analysisType') == analysisType
            }
            actualAnalysisKeys = {
                (entry.get('motorId'), entry.get('caseId')) for entry in analysisEntries
            }
            if actualAnalysisKeys != expectedAnalysisKeys:
                errors.append(f'{view}/{analysisType} component index does not match source view')
            if any(not (analysisIndexPath.parent / entry['path']).is_dir() for entry in analysisEntries):
                errors.append(f'{view}/{analysisType} contains a broken component path')
            analysisWhole = _load(analysisWholePath)
            expectedWholeKeys = {
                (entry.get('motorId'), entry.get('parentCaseId'))
                for entry in wholeEntries
                if entry.get('analysisType') == analysisType
            }
            actualWholeKeys = {
                (entry.get('motorId'), entry.get('parentCaseId'))
                for entry in analysisWhole.get('cases', ())
            }
            if actualWholeKeys != expectedWholeKeys:
                errors.append(f'{view}/{analysisType} whole-motor index does not match source view')

    if componentKeysByView.get('point') != componentKeysByView.get('graph'):
        errors.append('point and graph component identities differ')
    if parentAnalysisByView.get('point') != parentAnalysisByView.get('graph'):
        errors.append('point and graph temporal labels differ')
    if not hierarchy:
        errors.append('assembly hierarchy catalog is missing')
    else:
        expectedLevels = [
            'systemType', 'assemblyType', 'assemblyId', 'componentType', 'componentInstance'
        ]
        if hierarchy.get('levels') != expectedLevels:
            errors.append('assembly hierarchy levels are invalid')
        for systemType, system in hierarchy.get('systems', {}).items():
            for assemblyType, assembly in system.get('assemblyTypes', {}).items():
                for view, analyses in assembly.get('views', {}).items():
                    for analysisType, paths in analyses.items():
                        for scope, relativePath in paths.items():
                            branchPath = root / relativePath
                            if not branchPath.is_file():
                                errors.append(
                                    f'missing hierarchy index {systemType}/{assemblyType}/'
                                    f'{view}/{analysisType}/{scope}: {branchPath}'
                                )
                                continue
                            branch = _load(branchPath)
                            for entry in branch.get('cases', ()):
                                if (
                                    entry.get('systemType') != systemType
                                    or entry.get('assemblyType') != assemblyType
                                    or entry.get('analysisType') != analysisType
                                ):
                                    errors.append(f'hierarchy branch contains a mismatched case: {branchPath}')
                                if not (branchPath.parent / entry.get('path', '')).is_dir():
                                    errors.append(f'hierarchy branch contains a broken path: {branchPath}')
    assets = [path for path in root.rglob('assets') if path.is_dir()]
    if assets:
        errors.append(f'unexpected assets directory: {assets[0]}')
    temporary = [path for path in root.rglob('*.tmp') if path.is_file()]
    if temporary:
        errors.append(f'unexpected temporary file: {temporary[0]}')
    for source in views.get('sourceArchives', ()):
        if not Path(source).is_file():
            errors.append(f'source archive is missing: {source}')

    if errors:
        raise ValueError('dataset validation failed:\n- ' + '\n- '.join(errors[:30]))
    return {
        'motorIds': sorted(motorIds),
        'parentCaseCount': len(parentCaseIds),
        'componentCasesPerView': len(componentKeysByView['point']),
        'wholeMotorCasesPerView': len(parentCaseIds),
        'npzCount': npzCount,
        'npzSizeGiB': round(npzBytes / 1024**3, 3),
        'transientSeriesFiles': seriesCount,
        'transientSeriesSizeGiB': round(seriesBytes / 1024**3, 3),
        'componentTypes': dict(sorted(typeCounts.items())),
        'parentCaseAnalysisTypes': dict(
            sorted(Counter(parentAnalysisByView['point'].values()).items())
        ),
        'systemTypes': sorted(hierarchy.get('systems', {})),
        'assemblyTypes': sorted(
            {
                assemblyType
                for system in hierarchy.get('systems', {}).values()
                for assemblyType in system.get('assemblyTypes', {})
            }
        ),
        'sourceArchiveCount': len(views.get('sourceArchives', ())),
        'assetsDirectories': 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate a multi-motor full-resolution dataset.')
    parser.add_argument('dataset_root', type=Path)
    arguments = parser.parse_args()
    print(json.dumps(validate(arguments.dataset_root), indent=2))


if __name__ == '__main__':
    main()

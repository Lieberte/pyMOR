#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mor.nn.data.motorFullResolutionDataset import _writeJson
from mor.nn.data.motorTransientDataset import (
    buildMotorTransientSeriesFromSta,
    writeMotorThermalAnalysisIndexes,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build complete fixed-mesh time series for transient motor STA cases.'
    )
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--parent-case', action='append', dest='parent_cases')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--plan', action='store_true')
    arguments = parser.parse_args()

    root = arguments.dataset.resolve()
    pointIndexPath = root / 'point' / 'dataset.json'
    graphIndexPath = root / 'graph' / 'dataset.json'
    pointIndex = _load(pointIndexPath)
    graphIndex = _load(graphIndexPath)
    selected = set(arguments.parent_cases or ())
    parents = sorted(
        {
            entry['parentCaseId']
            for entry in graphIndex.get('cases', ())
            if entry.get('analysisType') == 'transient'
            and (not selected or entry['parentCaseId'] in selected)
        }
    )
    unknown = selected - set(parents)
    if unknown:
        raise ValueError(f'selected parent cases are not transient dataset cases: {sorted(unknown)}')
    plans = []
    for parentCaseId in parents:
        graphEntries = [
            entry for entry in graphIndex['cases'] if entry['parentCaseId'] == parentCaseId
        ]
        pointEntries = [
            entry for entry in pointIndex['cases'] if entry['parentCaseId'] == parentCaseId
        ]
        manifest = _load(root / 'graph' / graphEntries[0]['path'] / 'manifest.json')
        source = manifest.get('metadata', {}).get('sourceSta')
        if not source:
            raise ValueError(f'transient case has no sourceSta: {parentCaseId}')
        pointsByInstance = {entry['componentInstance']: entry for entry in pointEntries}
        complete = all(
            entry.get('timeSeriesStored')
            and (root / 'graph' / entry['path'] / 'series.h5').is_file()
            and (
                root
                / 'point'
                / pointsByInstance[entry['componentInstance']]['path']
                / 'series.h5'
            ).is_file()
            for entry in graphEntries
        )
        plans.append(
            {
                'parentCaseId': parentCaseId,
                'sourceSta': source,
                'componentCount': len(graphEntries),
                'complete': complete,
            }
        )
    print(json.dumps({'event': 'plan', 'cases': plans}, indent=2), flush=True)
    if arguments.plan:
        return

    summaries = []
    for position, plan in enumerate(plans, start=1):
        parentCaseId = plan['parentCaseId']
        if plan['complete'] and not arguments.overwrite:
            print(
                json.dumps({'event': 'sequence-skipped', 'parentCaseId': parentCaseId}),
                flush=True,
            )
            continue
        graphEntries = [
            entry for entry in graphIndex['cases'] if entry['parentCaseId'] == parentCaseId
        ]
        pointEntries = [
            entry for entry in pointIndex['cases'] if entry['parentCaseId'] == parentCaseId
        ]
        print(
            json.dumps(
                {
                    'event': 'sequence-start',
                    'parentCaseId': parentCaseId,
                    'position': position,
                    'count': len(plans),
                }
            ),
            flush=True,
        )
        summary = buildMotorTransientSeriesFromSta(
            plan['sourceSta'],
            root / 'graph',
            root / 'point',
            graphEntries,
            pointEntries,
            overwrite=arguments.overwrite,
        )
        summaries.append({'parentCaseId': parentCaseId, **summary})
        _writeJson(graphIndexPath, graphIndex)
        _writeJson(pointIndexPath, pointIndex)
        print(
            json.dumps({'event': 'sequence-complete', 'parentCaseId': parentCaseId, **summary}),
            flush=True,
        )

    writeMotorThermalAnalysisIndexes(root)
    print(json.dumps({'event': 'complete', 'cases': summaries}, indent=2), flush=True)


if __name__ == '__main__':
    main()

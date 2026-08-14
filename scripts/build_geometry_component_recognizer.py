#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mor.nn.data import loadGeometryRecognitionDatasetRecords
from mor.nn.runtime import geometryComponentRecognizer, geometryDescriptorConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a geometry-only component recognizer from a graph motor dataset.'
    )
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--surface-samples', type=int, default=8192)
    parser.add_argument('--pair-samples', type=int, default=32768)
    arguments = parser.parse_args()

    records = loadGeometryRecognitionDatasetRecords(
        arguments.dataset, deduplicateByMeshUuid=True
    )
    recognizer = geometryComponentRecognizer(
        config=geometryDescriptorConfig(
            surfaceSampleCount=arguments.surface_samples,
            pairSampleCount=arguments.pair_samples,
        )
    ).fitRecords(records)
    recognizer.save(arguments.output)
    print(
        json.dumps(
            {
                'model': str(arguments.output.resolve()),
                'independentGeometries': len(records),
                'componentTypes': sorted(recognizer.prototypes),
                'prototypeCounts': {
                    label: len(items) for label, items in recognizer.prototypes.items()
                },
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()

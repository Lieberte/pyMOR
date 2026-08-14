#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mor.nn.data import loadGeometryRecognitionCase
from mor.nn.runtime import geometryComponentRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Predict an approximate component type from graph NPZ geometry.'
    )
    parser.add_argument('model', type=Path)
    parser.add_argument('geometry', type=Path)
    parser.add_argument('--top-k', type=int, default=3)
    arguments = parser.parse_args()
    if arguments.top_k <= 0:
        parser.error('--top-k must be positive')

    recognizer = geometryComponentRecognizer.load(arguments.model)
    case = loadGeometryRecognitionCase(arguments.geometry)
    prediction = recognizer.predict(case, topK=arguments.top_k)
    print(json.dumps(prediction.toDict(), indent=2))


if __name__ == '__main__':
    main()

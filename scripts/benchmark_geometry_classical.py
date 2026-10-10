#!/usr/bin/env python3
"""Compare classical geometry classifiers on the invariant descriptor."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mor.nn.data import loadGeometryRecognitionDatasetRecords
from mor.nn.runtime.geometryComponentRecognizer import (
    extractGeometryDescriptor,
    geometryDescriptorConfig,
)


class encodedXgbClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, *, randomState: int = 0):
        self.randomState = int(randomState)

    def fit(self, features, labels):
        from xgboost import XGBClassifier

        self.classes_ = np.unique(labels)
        encoded = np.searchsorted(self.classes_, labels)
        self.model_ = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.7,
            objective='multi:softprob',
            eval_metric='mlogloss',
            tree_method='hist',
            random_state=self.randomState,
            n_jobs=-1,
        )
        self.model_.fit(features, encoded)
        return self

    def predict_proba(self, features):
        return self.model_.predict_proba(features)

    def predict(self, features):
        return self.classes_[self.model_.predict(features).astype(np.int64)]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark classical geometry-only component classifiers.'
    )
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--minimum-class-count', type=int, default=2)
    parser.add_argument('--validation-per-class', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--target-known-acceptance', type=float, default=0.875)
    parser.add_argument('--seed', type=int, default=17)
    return parser.parse_args()


def _split(records, minimumClassCount: int, validationPerClass: int, seed: int):
    grouped: dict[str, list] = {}
    for record in records:
        grouped.setdefault(record.componentType, []).append(record)
    rng = np.random.default_rng(seed)
    train, validation, excluded = [], [], []
    for label in sorted(grouped):
        items = grouped[label]
        if len(items) < minimumClassCount or len(items) <= validationPerClass:
            excluded.extend(items)
            continue
        order = rng.permutation(len(items))
        validation.extend(items[index] for index in order[:validationPerClass])
        train.extend(items[index] for index in order[validationPerClass:])
    trainMeshes = {item.meshUuid for item in train}
    validationMeshes = {item.meshUuid for item in validation}
    if trainMeshes & validationMeshes:
        raise ValueError('meshUuid leakage detected in classical split')
    return tuple(train), tuple(validation), tuple(excluded)


def _features(records, config, featureNames=None):
    values = []
    labels = []
    meshUuids = []
    for record in records:
        descriptor = extractGeometryDescriptor(record.load(), config)
        if featureNames is None:
            featureNames = descriptor.featureNames
        if descriptor.featureNames != featureNames:
            raise ValueError('geometry descriptor schemas do not match')
        values.append(descriptor.values)
        labels.append(record.componentType)
        meshUuids.append(record.meshUuid)
    if not values:
        return np.empty((0, len(featureNames or ()))), np.empty((0,), dtype=object), ()
    return np.stack(values), np.asarray(labels, dtype=object), tuple(meshUuids)


def _models(seed: int):
    models = {
        'svm_rbf': Pipeline(
            [
                ('scale', StandardScaler()),
                (
                    'model',
                    SVC(
                        C=3.0,
                        gamma='scale',
                        probability=True,
                        class_weight='balanced',
                        random_state=seed,
                    ),
                ),
            ]
        ),
        'svm_linear': Pipeline(
            [
                ('scale', StandardScaler()),
                (
                    'model',
                    SVC(
                        C=1.0,
                        kernel='linear',
                        probability=True,
                        class_weight='balanced',
                        random_state=seed,
                    ),
                ),
            ]
        ),
        'logistic_regression': Pipeline(
            [
                ('scale', StandardScaler()),
                ('model', LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced')),
            ]
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=250,
            max_features='sqrt',
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1,
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=250,
            max_features='sqrt',
            min_samples_leaf=1,
            class_weight='balanced_subsample',
            random_state=seed,
            n_jobs=-1,
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=2,
            random_state=seed,
        ),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            class_weight='balanced',
            random_state=seed,
        ),
        'knn': Pipeline(
            [
                ('scale', StandardScaler()),
                ('model', KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)),
            ]
        ),
        'shrinkage_lda': Pipeline(
            [
                ('scale', StandardScaler()),
                (
                    'model',
                    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                ),
            ]
        ),
        'gaussian_naive_bayes': Pipeline(
            [
                ('scale', StandardScaler()),
                ('model', GaussianNB(var_smoothing=1e-7)),
            ]
        ),
    }
    models['xgboost'] = encodedXgbClassifier(randomState=seed)
    return models


def _metrics(model, features, labels):
    probabilities = model.predict_proba(features)
    classNames = tuple(model.classes_)
    indices = np.argsort(probabilities, axis=1)[:, ::-1]
    topK = min(3, len(classNames))
    rankedLabels = np.asarray(classNames, dtype=object)[indices[:, :topK]]
    return {
        'top1': float(np.mean(rankedLabels[:, 0] == labels)),
        'top3': float(np.mean(np.any(rankedLabels == labels[:, None], axis=1))),
        'count': int(labels.size),
    }


def _calibration_threshold(model, xTrain, yTrain, targetAcceptance, seed):
    counts = Counter(yTrain)
    folds = min(3, min(counts.values()))
    if folds < 2:
        model.fit(xTrain, yTrain)
        probabilities = model.predict_proba(xTrain)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        probabilities = cross_val_predict(
            clone(model), xTrain, yTrain, cv=splitter, method='predict_proba'
        )
    scores = probabilities.max(axis=1)
    return float(np.quantile(scores, 1.0 - targetAcceptance, method='linear'))


def _aggregateRuns(runs):
    if not runs:
        return None
    keys = tuple(runs[0])
    aggregate = {}
    for key in keys:
        values = [run[key] for run in runs]
        if isinstance(values[0], (int, float)) and key not in ('count', 'knownCount', 'unknownCount'):
            aggregate[key] = {
                'mean': float(np.mean(values)),
                'standardDeviation': float(np.std(values)),
                'minimum': float(np.min(values)),
                'maximum': float(np.max(values)),
            }
        else:
            aggregate[key] = values[0]
    return aggregate


def _openSetMetrics(model, xKnown, yKnown, xUnknown, threshold):
    knownProbabilities = model.predict_proba(xKnown)
    unknownProbabilities = model.predict_proba(xUnknown)
    knownScores = knownProbabilities.max(axis=1)
    unknownScores = unknownProbabilities.max(axis=1)
    classNames = np.asarray(model.classes_, dtype=object)
    knownPredictions = classNames[np.argmax(knownProbabilities, axis=1)]
    knownCorrect = knownPredictions == yKnown
    knownAccepted = knownScores >= threshold
    unknownRejected = unknownScores < threshold
    return {
        'threshold': threshold,
        'knownTop1': float(np.mean(knownCorrect)),
        'knownAcceptance': float(np.mean(knownAccepted)),
        'knownCorrectAndAccepted': float(np.mean(knownCorrect & knownAccepted)),
        'unknownRejection': float(np.mean(unknownRejected)),
        'overall': float(
            (np.sum(knownCorrect & knownAccepted) + np.sum(unknownRejected))
            / (knownCorrect.size + unknownRejected.size)
        ),
        'knownCount': int(knownCorrect.size),
        'unknownCount': int(unknownRejected.size),
    }


def main() -> None:
    arguments = _arguments()
    if (
        arguments.minimum_class_count < 2
        or arguments.validation_per_class < 1
        or arguments.repeats < 1
        or not 0.0 < arguments.target_known_acceptance <= 1.0
    ):
        raise ValueError('split counts are invalid')
    records = loadGeometryRecognitionDatasetRecords(
        arguments.dataset, deduplicateByMeshUuid=True
    )
    config = geometryDescriptorConfig()
    firstDescriptor = extractGeometryDescriptor(records[0].load(), config)
    featureNames = firstDescriptor.featureNames
    xAll, yAll, allMeshes = _features(records, config, featureNames)
    indexByMesh = {meshUuid: index for index, meshUuid in enumerate(allMeshes)}
    if len(indexByMesh) != len(allMeshes):
        raise ValueError('deduplicated geometry records contain repeated meshUuid values')
    modelRuns = {name: {'closedSet': [], 'openSet': []} for name in _models(0)}
    firstSplit = None
    for repeat in range(arguments.repeats):
        splitSeed = arguments.seed + repeat * 104729
        trainRecords, validationRecords, excludedRecords = _split(
            records,
            arguments.minimum_class_count,
            arguments.validation_per_class,
            splitSeed,
        )
        trainIndices = np.asarray([indexByMesh[item.meshUuid] for item in trainRecords])
        validationIndices = np.asarray(
            [indexByMesh[item.meshUuid] for item in validationRecords]
        )
        unknownIndices = np.asarray(
            [indexByMesh[item.meshUuid] for item in excludedRecords]
        )
        xTrain, yTrain = xAll[trainIndices], yAll[trainIndices]
        xValidation, yValidation = xAll[validationIndices], yAll[validationIndices]
        xUnknown = xAll[unknownIndices]
        classNames = tuple(sorted(set(yTrain.tolist())))
        if firstSplit is None:
            firstSplit = {
                'seed': splitSeed,
                'train': [item.meshUuid for item in trainRecords],
                'validation': [item.meshUuid for item in validationRecords],
                'unknown': [item.meshUuid for item in excludedRecords],
            }
        for name, model in _models(splitSeed).items():
            model.fit(xTrain, yTrain)
            modelRuns[name]['closedSet'].append(
                _metrics(model, xValidation, yValidation)
            )
            if xUnknown.shape[0] > 0:
                threshold = _calibration_threshold(
                    model,
                    xTrain,
                    yTrain,
                    arguments.target_known_acceptance,
                    splitSeed,
                )
                modelRuns[name]['openSet'].append(
                    _openSetMetrics(
                        model,
                        xValidation,
                        yValidation,
                        xUnknown,
                        threshold,
                    )
                )
    results = {
        'formatVersion': 'geometry-classical-benchmark-v1',
        'seed': arguments.seed,
        'repeats': arguments.repeats,
        'featureCount': len(featureNames),
        'classNames': list(classNames),
        'trainCount': int(yTrain.size),
        'validationCount': int(yValidation.size),
        'excludedCount': int(xUnknown.shape[0]),
        'trainClassCounts': dict(sorted(Counter(yTrain).items())),
        'validationClassCounts': dict(sorted(Counter(yValidation).items())),
        'targetKnownAcceptance': arguments.target_known_acceptance,
        'models': {},
    }
    for name, runs in modelRuns.items():
        results['models'][name] = {
            'closedSet': _aggregateRuns(runs['closedSet']),
            'openSet': _aggregateRuns(runs['openSet']),
            'runs': runs,
        }
    results['firstSplit'] = firstSplit
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

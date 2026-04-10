import numpy as np

def toFloatArray(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=float)

def as2dFloatArray(points: np.ndarray, dim: int | None = None) -> np.ndarray:
    arr = toFloatArray(points)
    if arr.ndim == 0:
        raise ValueError('points must have at least one dimension')
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError('points must be 2D')
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(f'points must have dimension {dim}')
    return arr

def validateNodes(nodes: np.ndarray) -> np.ndarray:
    arr = as2dFloatArray(nodes)
    if arr.shape[0] == 0:
        raise ValueError('nodes must not be empty')
    return arr

def validateSampleCount(n: int) -> int:
    count = int(n)
    if count <= 0:
        raise ValueError('n must be positive')
    return count

def computeBoundingBox(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = validateNodes(nodes)
    return arr.min(axis=0).copy(), arr.max(axis=0).copy()

def computeCharacteristicLength(nodes: np.ndarray) -> float:
    arr = validateNodes(nodes)
    mn, mx = computeBoundingBox(arr)
    return float(np.max(mx - mn))

def mergeBoundaryTarget(
    target: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    dim: int | None = None,
) -> np.ndarray:
    if isinstance(target, (list, tuple)):
        if not target:
            if dim is None:
                raise ValueError('dim is required for empty boundary groups')
            return np.empty((0, dim), dtype=float)
        arrays = [as2dFloatArray(item, dim=dim) for item in target]
        if len(arrays) == 1:
            return arrays[0]
        return np.concatenate(arrays, axis=0)
    return as2dFloatArray(target, dim=dim)

def normalizeBoundaryMap(
    boundaryNodes: dict[str, np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]] | None,
    dim: int,
) -> dict[str, np.ndarray]:
    if not boundaryNodes:
        return {}
    normalized: dict[str, np.ndarray] = {}
    for name, target in boundaryNodes.items():
        normalized[name] = mergeBoundaryTarget(target, dim=dim)
    return normalized

def mergeBoundaryMap(boundaryNodes: dict[str, np.ndarray], dim: int) -> np.ndarray:
    if not boundaryNodes:
        return np.empty((0, dim), dtype=float)
    groups = [group for group in boundaryNodes.values() if group.size > 0]
    if not groups:
        return np.empty((0, dim), dtype=float)
    if len(groups) == 1:
        return groups[0]
    return np.concatenate(groups, axis=0)

def sampleRows(data: np.ndarray, n: int) -> np.ndarray:
    arr = as2dFloatArray(data)
    count = validateSampleCount(n)
    if arr.shape[0] == 0:
        raise ValueError('cannot sample from empty array')
    replace = count > arr.shape[0]
    idx = np.random.choice(arr.shape[0], count, replace=replace)
    return arr[idx]

def normalizeSampleWeights(weights: np.ndarray) -> np.ndarray:
    arr = toFloatArray(weights).reshape(-1)
    if arr.size == 0:
        raise ValueError('weights must not be empty')
    if np.any(arr < 0):
        raise ValueError('weights must be non-negative')
    total = arr.sum()
    if total <= 0:
        raise ValueError('weights must sum to a positive value')
    return arr / total

def splitSampleCounts(n: int, weights: np.ndarray) -> np.ndarray:
    count = validateSampleCount(n)
    normalized = normalizeSampleWeights(weights)
    raw = normalized * count
    counts = np.floor(raw).astype(int)
    remainder = count - counts.sum()
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts

def normalizeSampleCountMap(sampleCountByName: dict[str, int], emptyError: str) -> dict[str, int]:
    normalized = {name: int(count) for name, count in sampleCountByName.items() if int(count) > 0}
    if not normalized:
        raise ValueError(emptyError)
    return normalized

def splitSampleWeightMap(
    n: int,
    sampleWeightByName: dict[str, float],
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    names = list(sampleWeightByName.keys())
    weights = normalizeSampleWeights(np.asarray([sampleWeightByName[name] for name in names], dtype=float))
    counts = splitSampleCounts(n, weights)
    sampleCountByName = {name: int(count) for name, count in zip(names, counts)}
    return names, weights, sampleCountByName

def expandSampleWeights(
    names: list[str],
    sampleCountByName: dict[str, int],
    weights: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [np.full(sampleCountByName[name], weights[i], dtype=float) for i, name in enumerate(names) if sampleCountByName[name] > 0],
        axis=0,
    )

def fitUnitCubeTransform(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mn, mx = computeBoundingBox(nodes)
    span = mx - mn
    span = np.where(span < 1e-15, 1.0, span)
    return mn.astype(float), span.astype(float)

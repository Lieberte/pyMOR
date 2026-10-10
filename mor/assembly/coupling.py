from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


def _trace(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real')
    result = csr_matrix(value, dtype=float, copy=True)
    if min(result.shape) < 1 or not np.all(np.isfinite(result.data)) or np.any(result.data < 0):
        raise ValueError(f'{name} must be a nonempty finite non-negative trace matrix')
    if not np.allclose(np.asarray(result.sum(axis=1)).ravel(), 1., rtol=0., atol=1e-12):
        raise ValueError(f'{name} rows must sum to one to preserve constant temperatures')
    return result


@dataclass(frozen=True)
class thermalCouplingInterface:
    name: str
    primaryComponent: str
    secondaryComponent: str
    primaryTrace: object
    secondaryTrace: object
    conductance: object

    def __post_init__(self):
        if not all(isinstance(value, str) and value for value in
                   (self.name, self.primaryComponent, self.secondaryComponent)):
            raise ValueError('interface and component names must be nonempty strings')
        if self.primaryComponent == self.secondaryComponent:
            raise ValueError('an interface must connect two different components')
        primary, secondary = _trace(self.primaryTrace, 'primaryTrace'), _trace(self.secondaryTrace, 'secondaryTrace')
        if primary.shape[0] != secondary.shape[0]:
            raise ValueError('interface traces must use the same quadrature points')
        if np.iscomplexobj(self.conductance):
            raise ValueError('interface conductance must be real')
        conductance = np.broadcast_to(np.asarray(self.conductance, dtype=float), (primary.shape[0],)).copy()
        if not np.all(np.isfinite(conductance)) or np.any(conductance < 0):
            raise ValueError('interface conductance must be finite and non-negative (W/K)')
        for trace in (primary, secondary):
            for array in (trace.data, trace.indices, trace.indptr):
                array.setflags(write=False)
        conductance.setflags(write=False)
        object.__setattr__(self, 'primaryTrace', primary)
        object.__setattr__(self, 'secondaryTrace', secondary)
        object.__setattr__(self, 'conductance', conductance)

    @classmethod
    def fromNodePairs(cls, name, primaryComponent, secondaryComponent, primaryOrder, secondaryOrder,
                      primaryNodes, secondaryNodes, *, coefficient, areas):
        traces = []
        for order, nodes in ((primaryOrder, primaryNodes), (secondaryOrder, secondaryNodes)):
            indices = np.asarray(nodes)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not indices.size:
                raise ValueError('interface nodes must be nonempty one-dimensional integer arrays')
            if np.any(indices < 0) or np.any(indices >= order):
                raise ValueError('interface node index out of range')
            traces.append(csr_matrix((np.ones(indices.size), (np.arange(indices.size), indices)), shape=(indices.size, order)))
        if traces[0].shape[0] != traces[1].shape[0]:
            raise ValueError('primary and secondary node counts must match')
        area = np.broadcast_to(np.asarray(areas, dtype=float), (traces[0].shape[0],))
        htc = np.broadcast_to(np.asarray(coefficient, dtype=float), area.shape)
        if not np.all(np.isfinite(area)) or np.any(area <= 0):
            raise ValueError('interface areas must be positive and finite (m2)')
        if not np.all(np.isfinite(htc)) or np.any(htc < 0):
            raise ValueError('interface coefficient must be finite and non-negative (W/m2/K)')
        return cls(name, primaryComponent, secondaryComponent, *traces, htc * area)

    @classmethod
    def fromNearestNodes(cls, name, primaryComponent, secondaryComponent, primaryPoints, secondaryPoints,
                         primaryNodes, secondaryNodes, *, coefficient, areas, maxDistance):
        primary, secondary = np.asarray(primaryPoints, dtype=float), np.asarray(secondaryPoints, dtype=float)
        if (primary.ndim != 2 or secondary.ndim != 2 or primary.shape[1] != secondary.shape[1]
                or not np.all(np.isfinite(primary)) or not np.all(np.isfinite(secondary))):
            raise ValueError('interface points must be finite matrices with matching spatial dimensions')
        if not np.isfinite(maxDistance) or maxDistance <= 0:
            raise ValueError('maxDistance must be positive and finite')
        pNodes, sNodes = np.asarray(primaryNodes), np.asarray(secondaryNodes)
        for nodes, points in ((pNodes, primary), (sNodes, secondary)):
            if (nodes.ndim != 1 or not nodes.size or not np.issubdtype(nodes.dtype, np.integer)
                    or np.any(nodes < 0) or np.any(nodes >= len(points))):
                raise ValueError('interface nodes must be nonempty valid integer indices')
        distances, matches = cKDTree(secondary[sNodes]).query(primary[pNodes])
        if np.any(distances > maxDistance):
            raise ValueError(f'interface mapping exceeds maxDistance: {float(distances.max())}')
        return cls.fromNodePairs(name, primaryComponent, secondaryComponent, len(primary), len(secondary),
                                 pNodes, sNodes[matches], coefficient=coefficient, areas=areas)


@dataclass(frozen=True)
class projectedThermalInterface:
    name: str
    primaryComponent: str
    secondaryComponent: str
    primaryTrace: np.ndarray
    secondaryTrace: np.ndarray
    temperatureOffset: np.ndarray
    conductance: np.ndarray
    primaryFixedWeights: np.ndarray
    secondaryFixedWeights: np.ndarray

    def __post_init__(self):
        for field in ('primaryTrace', 'secondaryTrace', 'temperatureOffset', 'conductance',
                      'primaryFixedWeights', 'secondaryFixedWeights'):
            value = getattr(self, field)
            if np.iscomplexobj(value):
                raise ValueError(f'{field} must be real')
            array = np.array(value, dtype=float, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, field, array)


def projectThermalInterface(interface, primary, secondary):
    if interface.primaryTrace.shape[1] != primary.fullOrder or interface.secondaryTrace.shape[1] != secondary.fullOrder:
        raise ValueError('interface trace column counts must match the component full orders')
    return projectedThermalInterface(
        interface.name, interface.primaryComponent, interface.secondaryComponent,
        interface.primaryTrace @ primary.basis, interface.secondaryTrace @ secondary.basis,
        interface.primaryTrace @ primary.offset - interface.secondaryTrace @ secondary.offset,
        interface.conductance.copy(), interface.primaryTrace @ primary.fixedMask,
        interface.secondaryTrace @ secondary.fixedMask,
    )


def addProjectedThermalInterface(K, load, interface, primarySlice, secondarySlice, batchSize=4096):
    for start in range(0, len(interface.conductance), batchSize):
        rows = slice(start, start + batchSize)
        p, s, g = interface.primaryTrace[rows], interface.secondaryTrace[rows], interface.conductance[rows]
        cross = p.T @ (g[:, None] * s)
        K[primarySlice, primarySlice] += p.T @ (g[:, None] * p)
        K[secondarySlice, secondarySlice] += s.T @ (g[:, None] * s)
        K[primarySlice, secondarySlice] -= cross
        K[secondarySlice, primarySlice] -= cross.T
        source = g * interface.temperatureOffset[rows]
        load[primarySlice] -= p.T @ source
        load[secondarySlice] += s.T @ source

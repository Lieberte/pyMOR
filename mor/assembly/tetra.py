"""NumPy linear tetrahedron element kernel; no global matrix or backend allocation."""
import numpy as np


def thermalElementMatrices(coordinates, conductivity, density, specificHeat, heatSource,
                           capacityMode, startIndex=0):
    reference = np.array([[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    jacobians = np.stack([coordinates[:, 1] - coordinates[:, 0],
                         coordinates[:, 2] - coordinates[:, 0],
                         coordinates[:, 3] - coordinates[:, 0]], axis=2)
    volumes = np.abs(np.linalg.det(jacobians)) / 6.
    degenerate = np.flatnonzero(volumes <= np.finfo(float).eps)
    if degenerate.size:
        raise ValueError(f'degenerate tetra element at index {startIndex + int(degenerate[0])}')
    gradients = np.einsum('ai,eij->eaj', reference, np.linalg.inv(jacobians))
    stiffness = volumes[:, None, None] * np.einsum('eai,eij,ebj->eab', gradients, conductivity, gradients)
    heatCapacity = density * specificHeat
    template = ((np.ones((4, 4)) + np.eye(4)) / 20. if capacityMode == 'consistent'
                else np.eye(4) / 4.)
    capacity = heatCapacity[:, None, None] * volumes[:, None, None] * template[None, :, :]
    load = heatSource * volumes / 4.
    return volumes, stiffness, capacity, load

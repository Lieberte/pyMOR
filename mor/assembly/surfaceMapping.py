"""Closest-point traces for nonmatching triangular thermal interfaces."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


def closestTriangleTrace(points, nodes, triangles):
    """Return barycentric surface traces and distances, without extending a radius.

    Bounding spheres select candidates; exact plane/edge projections find the
    closest point. The sphere search includes every potentially closer triangle.
    """
    points, nodes = np.asarray(points, float), np.asarray(nodes, float)
    triangles = np.asarray(triangles)
    if (points.ndim != 2 or nodes.ndim != 2 or points.shape[1] != 3 or nodes.shape[1] != 3
            or not np.all(np.isfinite(points)) or not np.all(np.isfinite(nodes))):
        raise ValueError('points and nodes must be finite (n, 3) arrays')
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer)
            or np.any(triangles < 0) or np.any(triangles >= len(nodes))):
        raise ValueError('triangles must contain valid integer node indices')
    xyz = nodes[triangles]
    centers = xyz.mean(axis=1)
    radii = np.linalg.norm(xyz - centers[:, None], axis=2).max(axis=1)
    edge1, edge2 = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    normal = np.cross(edge1, edge2)
    if np.any(np.linalg.norm(normal, axis=1) == 0):
        raise ValueError('surface contains degenerate triangles')
    tree = cKDTree(centers)
    centerDistance, nearest = tree.query(points)
    # A point on the nearest-center triangle is at most d_center + radius
    # away. Add the largest candidate radius for a complete sphere search.
    candidates = tree.query_ball_point(points, centerDistance + radii[nearest] + radii.max() + 1e-14)
    indices = np.empty((len(points), 3), int)
    weights = np.empty((len(points), 3))
    distances = np.empty(len(points))
    for row, (point, ids) in enumerate(zip(points, candidates)):
        ids = np.asarray(ids, int)
        a, b = edge1[ids], edge2[ids]
        delta = point - xyz[ids, 0]
        aa, ab, bb = (a*a).sum(1), (a*b).sum(1), (b*b).sum(1)
        ad, bd = (a*delta).sum(1), (b*delta).sum(1)
        determinant = aa*bb - ab*ab
        u, v = (bb*ad-ab*bd)/determinant, (aa*bd-ab*ad)/determinant
        barycentric = np.column_stack((1-u-v, u, v))
        projected = np.einsum('ni,nij->nj', barycentric, xyz[ids])
        distance2 = ((projected-point)**2).sum(1)
        distance2[np.any(barycentric < 0, axis=1)] = np.inf
        for start, end in ((0, 1), (1, 2), (2, 0)):
            edge = xyz[ids, end] - xyz[ids, start]
            fraction = np.clip(((point-xyz[ids, start])*edge).sum(1)/(edge*edge).sum(1), 0, 1)
            projected = xyz[ids, start] + fraction[:, None]*edge
            candidate2 = ((projected-point)**2).sum(1)
            better = candidate2 < distance2
            distance2[better] = candidate2[better]
            barycentric[better] = 0
            barycentric[better, start] = 1-fraction[better]
            barycentric[better, end] = fraction[better]
        best = np.argmin(distance2)
        indices[row], weights[row] = triangles[ids[best]], barycentric[best]
        distances[row] = np.sqrt(distance2[best])
    weights = np.maximum(weights, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    trace = csr_matrix((weights.ravel(), (np.repeat(np.arange(len(points)), 3), indices.ravel())),
                       shape=(len(points), len(nodes)))
    return trace, distances

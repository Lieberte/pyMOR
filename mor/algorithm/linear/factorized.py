def factorizedSolve(data, backend):
    """Compatibility entry; reusable numerical factors belong to the backend."""
    return backend.linalg.factorize(data)

def selectDMDVariant(backendName: str = 'scipy', **kwargs) -> str:
    requests = []
    if kwargs.get('controlSnapshots') is not None:
        requests.append(('dmdc', 'controlSnapshots'))
    if kwargs.get('delay') is not None:
        requests.append(('hankel', 'delay'))
    if kwargs.get('noiseAware', False):
        requests.append(('tls', 'noiseAware'))
    if kwargs.get('bidirectional', False):
        requests.append(('forwardBackward', 'bidirectional'))
    if kwargs.get('optimize', False):
        requests.append(('optimized', 'optimize'))
    if len(requests) > 1:
        options = ', '.join(option for _, option in requests)
        raise ValueError(f"DMD auto-selection received incompatible requests: {options}.")
    variant = requests[0][0] if requests else 'exact'
    if backendName != 'scipy' and variant != 'exact':
        raise NotImplementedError(
            f"DMD variant '{variant}' is not verified for backend '{backendName}'."
        )
    return variant


selectDmdVariant = selectDMDVariant

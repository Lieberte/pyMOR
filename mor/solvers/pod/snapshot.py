from .base import podSolverBase
from mor.solvers.registry import solverRegistry

class snapshotPodSolver(podSolverBase):
    pass

solverRegistry.register('pod', 'snapshot', snapshotPodSolver)

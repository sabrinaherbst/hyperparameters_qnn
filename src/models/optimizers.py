# defines optimizers to be used

from qiskit.algorithms.optimizers import COBYLA, NELDER_MEAD, SPSA

def get_cobyla(maxiter=1000, tol=None):
    """
    returns a COBYLA optimizer
    """
    return COBYLA(maxiter=maxiter, tol=tol)

def get_spsa(maxiter=100, tol=None):
    """
    returns a SPSA optimizer
    """
    return SPSA(maxiter=maxiter)

def get_neldermead(maxiter=None, tol=None, adaptive=True):
    """
    returns a NELDER_MEAD optimizer
    """
    return NELDER_MEAD(maxiter=maxiter, tol=tol, adaptive=adaptive)

# contains code implementing different quantum data encoding schemes

from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap

def get_zfeaturemap(feature_dimension = 7, reps = 2):
    """
    returns a ZFeatureMap circuit
    """
    return ZFeatureMap(feature_dimension=feature_dimension, reps=reps)

def get_zzfeaturemap(feature_dimension = 7, reps = 2, entanglement = 'full'):
    """
    returns a ZZFeatureMap circuit
    """
    return ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement=entanglement)

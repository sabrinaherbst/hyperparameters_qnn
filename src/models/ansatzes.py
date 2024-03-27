# defines ansatzes to be used

from qiskit.circuit.library import TwoLocal, PauliTwoDesign, RealAmplitudes, EfficientSU2, ExcitationPreserving

def get_twolocal(num_qubits=7, rotation_blocks=['ry'], entanglement_blocks=['cx'], entanglement='full', reps=3):
    """
    returns a TwoLocal circuit
    """
    return TwoLocal(num_qubits=num_qubits, rotation_blocks=rotation_blocks, entanglement_blocks=entanglement_blocks, entanglement=entanglement, reps=reps)

def get_paulitwodesign(num_qubits=7, reps=3, seed=3):
    """
    returns a PauliTwoDesign circuit
    """
    return PauliTwoDesign(num_qubits=num_qubits, reps=reps, seed=seed)

def get_realamplitudes(num_qubits=7, reps=3, entanglement='reverse_linear'):
    """
    returns a RealAmplitudes circuit
    """
    return RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)

def get_efficientsu2(num_qubits=7, reps=3, su2_gates=None, entanglement='reverse_linear'):
    """
    returns a EfficientSU2 circuit
    """
    return EfficientSU2(num_qubits=num_qubits, reps=reps, su2_gates=su2_gates, entanglement=entanglement)
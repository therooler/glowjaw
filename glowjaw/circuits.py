import numpy as np
from typing import List, Tuple, Union
from glowjaw.ops import assert_state, identity
from glowjaw.gates import single_qubit_gate, two_qubit_gate
from glowjaw.observables import single_qubit_observable, two_qubit_observable
from glowjaw.channels import kraus_dep



def first_neighbours_indices2(control, target):
    ind = []
    distance = abs(control - target)

    if control < target:
        for i in range(distance - 1):
            ind.append([control + i + 1, control + i])
            ind.append([control + i, control + i + 1])
        ind.append([control + distance - 1, target])
        for i in range(distance - 2, -1, -1):
            ind.append([control + i, control + i + 1])
            ind.append([control + i + 1, control + i])
    elif control > target:
        for i in range(distance - 1):
            ind.append([control - i - 1, control - i])
            ind.append([control - i, control - i - 1])
        ind.append([control - distance + 1, target])
        for i in range(distance - 2, -1, -1):
            ind.append([control - i, control - i - 1])
            ind.append([control - i - 1, control - i])
    return (ind)


def tfi_ideal(params_np, config):
    depth = config['depth']
    nspins = config['nspins']

    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    # gates observables
    I = np.eye(2, 2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    ZZ = np.kron(Z, Z).reshape([2] * 4)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    RZ = lambda theta: np.cos(theta) * I - 1j * np.sin(theta) * Z
    RX = lambda theta: np.cos(theta) * I - 1j * np.sin(theta) * X
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    CNOT = CNOT.reshape([2] * 4)
    # initialize all qubits in ket0
    psi0 = identity(config['nspins'])
    for n in range(config['nspins']):
        psi0 = single_qubit_gate(psi0, hadamard, n)
    cnot_count = 0
    for d in range(depth):
        # apply Z gate
        for label in zip(range(config['nspins']), range(1, nspins)):
            psi0 = two_qubit_gate(psi0, CNOT, label)
            cnot_count += 1
            psi0 = single_qubit_gate(psi0, RZ(params_np[d] / 2), label[1])
            psi0 = two_qubit_gate(psi0, CNOT, label)
            cnot_count += 1
        psi0 = two_qubit_gate(psi0, CNOT, [config['nspins'] - 1, 0])
        cnot_count += 1
        psi0 = single_qubit_gate(psi0, RZ(params_np[d] / 2), 0)
        psi0 = two_qubit_gate(psi0, CNOT, [config['nspins'] - 1, 0])
        cnot_count += 1
        for n in range(config['nspins']):
            psi0 = single_qubit_gate(psi0, RX(params_np[d + depth] / 2), n)
    print(f"CNOT count = {cnot_count}")
    e = 0
    for label in zip(range(config['nspins']), range(1, nspins)):
        e -= two_qubit_observable(psi0, ZZ, label)
    e -= two_qubit_observable(psi0, ZZ, [config['nspins'] - 1, 0])
    for n in range(config['nspins']):
        e -= config['g'] * single_qubit_observable(psi0, X, n)
    return e, psi0


def tfi_noisy(params_np, config):
    depth = config['depth']
    nspins = config['nspins']
    K = kraus_dep(config['noise_unit'])

    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    # gates observables
    I = np.eye(2, 2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    ZZ = np.kron(Z, Z).reshape([2] * 4)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    RZ = lambda theta: np.cos(theta) * I - 1j * np.sin(theta) * Z
    RX = lambda theta: np.cos(theta) * I - 1j * np.sin(theta) * X
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    CNOT = CNOT.reshape([2] * 4)
    # initialize all qubits in ket0
    psi0 = identity(nspins)

    for n in range(nspins):
        psi0 = single_qubit_gate(psi0, hadamard, n)
    cnot_count = 0
    for d in range(depth):
        # apply Z gate
        for label in zip(range(nspins), range(1, nspins)):
            psi0 = two_qubit_gate(psi0, CNOT, label)
            psi0 = single_qubit_channel(psi0, K, label[0])
            psi0 = single_qubit_channel(psi0, K, label[1])
            cnot_count += 1

            psi0 = single_qubit_gate(psi0, RZ(params_np[d] / 2), label[1])

            psi0 = two_qubit_gate(psi0, CNOT, label)
            psi0 = single_qubit_channel(psi0, K, label[0])
            psi0 = single_qubit_channel(psi0, K, label[1])
            cnot_count += 1
        for label in first_neighbours_indices2(0, nspins - 1):
            psi0 = two_qubit_gate(psi0, CNOT, label)
            psi0 = single_qubit_channel(psi0, K, label[0])
            psi0 = single_qubit_channel(psi0, K, label[1])
            cnot_count += 1

        psi0 = single_qubit_gate(psi0, RZ(params_np[d] / 2), nspins - 1)

        for label in first_neighbours_indices2(0, nspins - 1):
            psi0 = two_qubit_gate(psi0, CNOT, label)
            psi0 = single_qubit_channel(psi0, K, label[0])
            psi0 = single_qubit_channel(psi0, K, label[1])
            cnot_count += 1

        for n in range(nspins):
            psi0 = single_qubit_gate(psi0, RX(params_np[d + depth] / 2), n)

    print(f"CNOT count = {cnot_count}")
    e = 0
    for label in zip(range(nspins), range(1, nspins)):
        e -= two_qubit_observable(psi0, ZZ, label)
    e -= two_qubit_observable(psi0, ZZ, [nspins - 1, 0])
    for n in range(nspins):
        e -= config['g'] * single_qubit_observable(psi0, X, n)
    return e, psi0



def circuit_gradient(circuit, params):
    grad = []
    params_shape = params.shape
    params = params.flatten()
    for i in range(params.size):
        params_np_plus = np.copy(params)
        params_np_plus[i] += np.pi / 2
        params_np_min = np.copy(params)
        params_np_min[i] -= np.pi / 2
        grad.append(0.5 * (circuit(params_np_plus.reshape(params_shape)) - circuit(params_np_min.reshape(params_shape))))
    return np.array(grad).reshape(params_shape)
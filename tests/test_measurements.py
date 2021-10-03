import numpy as np

from glowjaw.ops import identity, to_matrix, X,Y,Z, H,CNOT, RY
from glowjaw.gates import single_qubit_gate, two_qubit_gate
from glowjaw.observables import single_qubit_observable, two_qubit_observable
from glowjaw.channels import single_qubit_projector,single_qubit_projector_probability

import pytest
import cirq

import itertools as it

def test_measure_xyz_observable_zero_state():
    """Creating |0> and measuring X,Y,Z"""
    N = 1
    rho = identity(N)
    expval_x = single_qubit_observable(rho, X, 0)
    expval_y = single_qubit_observable(rho, Y, 0)
    expval_z = single_qubit_observable(rho, Z, 0)
    assert expval_x == 0j, f'expected `expval_x` to be 0, received {expval_x}'
    assert expval_y == 0j, f'expected `expval_y` to be 0, received {expval_y}'
    assert expval_z == 1 + 0j, f'expected `expval_z` to be 1, received {expval_z}'


def test_measure_xyz_observable_one_state():
    """Creating |1> and measuring X,Y,Z"""
    N = 1
    rho = identity(N)
    rho = single_qubit_gate(rho, X, 0)
    expval_x = single_qubit_observable(rho, X, 0)
    expval_y = single_qubit_observable(rho, Y, 0)
    expval_z = single_qubit_observable(rho, Z, 0)
    assert np.isclose(expval_x, 0), f'expected `expval_x` to be 0, received {expval_x}'
    assert np.isclose(expval_y, 0), f'expected `expval_y` to be 0, received {expval_y}'
    assert np.isclose(expval_z, -1), f'expected `expval_z` to be -1, received {expval_z}'


def test_measure_xyz_observable_bellstate():
    """Creating |Bell+> and measuring X,Y,Z on both qubits"""
    N = 2
    rho = identity(N)
    rho = single_qubit_gate(rho, H, 0)
    rho = two_qubit_gate(rho, CNOT, [0, 1])
    expval_x1 = single_qubit_observable(rho, X, 0)
    expval_y1 = single_qubit_observable(rho, Y, 0)
    expval_z1 = single_qubit_observable(rho, Z, 0)
    expval_x2 = single_qubit_observable(rho, X, 1)
    expval_y2 = single_qubit_observable(rho, Y, 1)
    expval_z2 = single_qubit_observable(rho, Z, 1)
    assert np.isclose(expval_x1, 0.), f'expected `expval_x` to be 0, received {expval_x1}'
    assert np.isclose(expval_y1, 0.), f'expected `expval_y` to be 0, received {expval_y1}'
    assert np.isclose(expval_z1, 0.), f'expected `expval_z` to be 0, received {expval_z1}'
    assert np.isclose(expval_x2, 0.), f'expected `expval_x` to be 0, received {expval_x2}'
    assert np.isclose(expval_y2, 0.), f'expected `expval_y` to be 0, received {expval_y2}'
    assert np.isclose(expval_z2, 0.), f'expected `expval_z` to be 0, received {expval_z2}'


def test_measure_xx_yy_zz_observable_bellstate():
    """Creating |Bell+> and measuring XX,YY,ZZ on both qubits"""
    N = 2
    rho = identity(N)
    rho = single_qubit_gate(rho, H, 0)
    rho = two_qubit_gate(rho, CNOT, [0, 1])
    expval_xx = two_qubit_observable(rho, np.kron(X, X).reshape([2] * 4), (0, 1))
    expval_yy = two_qubit_observable(rho, np.kron(Y, Y).reshape([2] * 4), (0, 1))
    expval_zz = two_qubit_observable(rho, np.kron(Z, Z).reshape([2] * 4), (0, 1))
    assert np.isclose(expval_xx, 1.0), f'expected `expval_xx` to be 1, received {expval_xx}'
    assert np.isclose(expval_yy, -1.0), f'expected `expval_yy` to be -1, received {expval_yy}'
    assert np.isclose(expval_zz, 1.0), f'expected `expval_zz` to be 1, received {expval_zz}'


@pytest.mark.parametrize('N, number_of_gates', it.product([2, 4, 6, 8], [5, 15, 25]))
def test_random_nn_cnot_circuit_cirq_comparison(N: int, number_of_gates: int):
    """Creating random circuit with hadamard on all qubits and random nearest neighbour CNOTs
    and measuring X, Y, Z and neareset neighbour XX,YY,ZZ on all qubits"""
    qubits = cirq.GridQubit.rect(1, N)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q) for q in qubits)
    angles = np.random.randn(N)
    circuit.append(cirq.Y(q)**(angles[i]) for i,q in enumerate(qubits))

    indices = np.random.randint(0, N - 1, number_of_gates)
    circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]) for i in indices)

    state = cirq.Simulator().simulate(circuit)._final_simulator_state
    print(state)
    ps = []
    for i in range(N - 1):
        ps.append(cirq.X(qubits[i]))
        ps.append(cirq.Y(qubits[i]))
        ps.append(cirq.Z(qubits[i]))
        ps.append(cirq.X(qubits[i]) * cirq.X(qubits[i + 1]))
        ps.append(cirq.Y(qubits[i]) * cirq.Y(qubits[i + 1]))
        ps.append(cirq.Z(qubits[i]) * cirq.Z(qubits[i + 1]))

    qubit_map = {qubits[i]: i for i in range(N)}
    expvals_cirq = [p.expectation_from_state_vector(state.state_vector, qubit_map).real for p in ps]
    rho = identity(N)
    for i in range(N):
        rho = single_qubit_gate(rho, H, i)
    for i in range(N):
        rho = single_qubit_gate(rho, RY(angles[i]*np.pi/2.), i)
    for i in indices:
        rho = two_qubit_gate(rho, CNOT, [i, i + 1])
    expvals_rho = []
    for i in range(N - 1):
        expvals_rho.append(single_qubit_observable(rho, X, i))
        expvals_rho.append(single_qubit_observable(rho, Y, i))
        expvals_rho.append(single_qubit_observable(rho, Z, i))
        expvals_rho.append(two_qubit_observable(rho, np.kron(X, X).reshape([2] * 4), (i, i + 1)))
        expvals_rho.append(two_qubit_observable(rho, np.kron(Y, Y).reshape([2] * 4), (i, i + 1)))
        expvals_rho.append(two_qubit_observable(rho, np.kron(Z, Z).reshape([2] * 4), (i, i + 1)))
    assert np.allclose(expvals_cirq, expvals_rho, atol=1e-5), \
        f'expected expectation values between the density matrix and cirq state have to be close,' \
        f' but received {expvals_rho} and {expvals_cirq}'


@pytest.mark.parametrize('N, number_of_gates', it.product([2, 4, 6, 8], [5, 15, 25]))
def test_random_angles_circuit_cirq_comparison(N: int, number_of_gates: int):
    """Creating random circuit with hadamard on all qubits and random Y rotations
    and measuring X, Y, Z and neareset neighbour XX,YY,ZZ on all qubits"""

    qubits = cirq.GridQubit.rect(1, N)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q) for q in qubits)
    angles = np.random.rand(number_of_gates)
    indices = np.random.randint(0, N, number_of_gates)
    circuit.append(cirq.Y(qubits[idx]) ** angles[i] for i, idx in enumerate(indices))

    state = cirq.Simulator().simulate(circuit)._final_simulator_state
    print(state)
    ps = []
    for i in range(N - 1):
        ps.append(cirq.X(qubits[i]))
        ps.append(cirq.Y(qubits[i]))
        ps.append(cirq.Z(qubits[i]))
        ps.append(cirq.X(qubits[i]) * cirq.X(qubits[i + 1]))
        ps.append(cirq.Y(qubits[i]) * cirq.Y(qubits[i + 1]))
        ps.append(cirq.Z(qubits[i]) * cirq.Z(qubits[i + 1]))

    qubit_map = {qubits[i]: i for i in range(N)}
    expvals_cirq = [p.expectation_from_state_vector(state.state_vector, qubit_map).real for p in ps]
    rho = identity(N)
    for i in range(N):
        rho = single_qubit_gate(rho, H, i)
    for i, idx in enumerate(indices):
        rho = single_qubit_gate(rho, RY(angles[i] * np.pi / 2), idx)
    expvals_rho = []
    for i in range(N - 1):
        expvals_rho.append(single_qubit_observable(rho, X, i))
        expvals_rho.append(single_qubit_observable(rho, Y, i))
        expvals_rho.append(single_qubit_observable(rho, Z, i))
        expvals_rho.append(two_qubit_observable(rho, np.kron(X, X).reshape([2] * 4), (i, i + 1)))
        expvals_rho.append(two_qubit_observable(rho, np.kron(Y, Y).reshape([2] * 4), (i, i + 1)))
        expvals_rho.append(two_qubit_observable(rho, np.kron(Z, Z).reshape([2] * 4), (i, i + 1)))
    assert np.allclose(expvals_cirq, expvals_rho, atol=1e-5), \
        f'expected expectation values between the density matrix and cirq state to be close,' \
        f' but received {expvals_rho} and {expvals_cirq}'


def test_projector_single_qubit_zero_state():
    """Testing Pi rho Pi / Z with Pi = |0><0| """
    N = 2
    rho = identity(N)
    pi = np.zeros((2, 2), complex)
    rho_true = np.copy(identity(N))
    pi[0, 0] = 1.0
    rho = single_qubit_gate(rho, H, 0)
    p = single_qubit_projector_probability(rho, pi, 0)
    assert np.isclose(p, 0.5), 'probability `p` must be close to 0.5 '
    rho = single_qubit_projector(rho, pi, 0)
    print(to_matrix(rho))
    print(to_matrix(rho_true))
    assert np.allclose(rho, rho_true, rtol=1e-5), '`rho` must be close to the projected state.'


def test_projector_single_qubit_one_state():
    """Testing Pi rho Pi / Z with Pi = |1><1| """
    N = 2
    rho = identity(N)
    pi = np.zeros((2, 2), complex)
    pi[1, 1] = 1.0
    p = single_qubit_projector_probability(rho, pi, 0)
    assert np.isclose(p, 0.0), 'probability `p` must be close to 0.0 '
    with pytest.warns(RuntimeWarning, match='Divide by zero encountered because projector is orthogonal'):
        single_qubit_projector(rho, pi, 0)


@pytest.mark.parametrize('N, number_of_gates', it.product([2, 4, 6, 8], [5, 15, 25]))
def test_projector_single_qubit_composed_observable_matches(N: int, number_of_gates: int):
    """'Testing if <Z> = sum_i p_i Tr{|i><i| rho} with i=0,1 and random product rho"""
    rho = identity(N)
    projectors = [np.zeros((2, 2), complex), np.zeros((2, 2), complex)]
    eigenvalues_z = [+1, -1]
    projectors[0][0, 0] = 1.
    projectors[1][1, 1] = 1.
    indices = np.random.randint(0, N - 1, number_of_gates)
    angles = np.random.rand(number_of_gates)
    for i in range(N):
        rho = single_qubit_gate(rho, H, i)
    for i in range(number_of_gates):
        rho = single_qubit_gate(rho, RY(angles[i] * np.pi / 2.), indices[i])
        rho = two_qubit_gate(rho, CNOT, (indices[i]+ 1, indices[i] ))

    expval_z = single_qubit_observable(rho, Z, 0)
    projector_total_expvals = 0
    for i in range(2):
        probs = single_qubit_projector_probability(rho, projectors[i], 0)
        projector_expvals = eigenvalues_z[i]
        projector_total_expvals += probs * projector_expvals
    assert np.isclose(expval_z, projector_total_expvals), 'Expectation value `expval_z` must be close' \
                                                          ' to `projector_total_expvals`'

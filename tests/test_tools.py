import numpy as np

from glowjaw.ops import identity, to_matrix, I, X,Y,Z, H,CNOT, RY, partial_trace, single_reprepare, trace
from glowjaw.gates import single_qubit_gate, two_qubit_gate
from glowjaw.observables import single_qubit_observable, two_qubit_observable
from glowjaw.channels import single_qubit_projector,single_qubit_projector_probability, two_qubit_effect

import pytest

def test_bellstate_2():
    """Creating 2 qubit Bell+ state"""
    N = 2
    rho = identity(N)
    rho = single_qubit_gate(rho, H, 0)
    rho = two_qubit_gate(rho, CNOT, [0, 1])
    rho = partial_trace(rho, [0, ])
    rho_true = np.eye(2, 2, dtype=complex) * 0.5
    assert np.allclose(rho_true, rho), f"expected `psi_true` to be equal to eye(2,2)*0.5, but obtained {rho}"


@pytest.mark.parametrize('site_rep', [0, 1, 2])
def test_repeparation_1(site_rep: int):
    """Trace out and reprepare"""
    N = 3
    rho = identity(N)
    for i in range(N):
        if i != site_rep:
            rho = single_qubit_gate(rho, H, i)

    rho_true = np.copy(rho)
    zero_state = np.zeros((2, 2), complex)
    zero_state[0, 0] = 1.0
    rho = single_reprepare(rho, zero_state, site_rep)
    assert np.allclose(rho_true, rho), f"expected `psi_true` to be equal to |bell+><bell+|, but obtained {rho}"


@pytest.mark.parametrize('N', [2, 4, 6])
def test_partial_trace(N: int):
    """Check that partial trace gives a trace one state"""
    RY = lambda theta: np.cos(theta) * I - 1j * np.sin(theta) * Y

    rho = identity(N)
    for i in range(25):
        qidx = np.random.randint(0, N)
        cnotidx = np.random.randint(0, N - 1)
        rho = single_qubit_gate(rho, H, qidx)
        rho = single_qubit_gate(rho, RY(np.random.rand(1)), qidx)
        rho = two_qubit_gate(rho, CNOT, [cnotidx, cnotidx + 1])
    rho = partial_trace(rho, [0, ])

    assert np.isclose(trace(rho),
                      1.0), f'partial trace must return a trace 1 state, received tr(rho) = {trace(rho)}'

#
# @pytest.mark.parametrize('N', [3, 4, 5])
# def test_repreparatation_2(N: int):
#     """Check that partial trace gives a trace one state"""
#     rho = random_mps(N, 2, 12)
#     rho_full = mps2op(rho)
#     transition = transition_mat()
#     a0, a3, b0, b3 = np.random.choice(list(range(6)), 4)
#     phi_quasi1 = reprepare2(rho, 0, transition[a3][a0], N - 1, transition[b3][b0])
#     phi_quasi1 = mps2op(phi_quasi1)
#
#     rho_full = two_qubit_effect(rho_full.reshape([2] * (2 * N)),
#                                        np.kron(transition[a3][a0], transition[b3][b0]).astype(complex).reshape(
#                                            (2, 2, 2, 2)), [0, N - 1])
#
#     assert np.allclose(to_matrix(rho_full),
#                        phi_quasi1), 'Expected MPS reprepare and full rho state to be close'
#
#
# @pytest.mark.parametrize('N', [3, 4, 5])
# def test_effect_probability(N: int):
#     """Check that partial trace gives a trace one state"""
#     rho = random_mps(N, 2, 12)
#     rho_full = mps2op(rho)
#     transition = transition_mat()
#     frame, pinvT, coin = pauli6_frame(1)
#     a0, a3, b0, b3 = np.random.choice(list(range(6)), 4)
#     p_mps = meas2(rho, 0, frame[a0], N - 1, frame[b0], )
#
#     p_full = two_qubit_effect_probability(rho_full.reshape([2] * (2 * N)),
#                                                  np.kron(frame[a0], frame[b0]).astype(complex).reshape((2, 2, 2, 2)),
#                                                  [0, N - 1])
#     print(p_full)
#     print(p_mps)

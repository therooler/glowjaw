import numpy as np

from typing import List, Tuple, Union

from glowjaw.ops import identity, to_matrix,I,X,H,CNOT, to_tensor
from glowjaw.gates import single_qubit_gate, two_qubit_gate
from glowjaw.observables import single_qubit_observable, two_qubit_observable

import pytest


def test_identity_rho():
    """'Testing U rho U^-1 with U = I (x) I '"""
    N = 2
    rho = identity(N)
    rho_true = np.copy(rho)
    rho_true = rho_true.reshape(2 ** N, 2 ** N)
    rho = single_qubit_gate(rho, I, 0)
    rho = single_qubit_gate(rho, I, 1)
    rho = rho.reshape(2 ** N, 2 ** N)
    assert np.allclose(rho_true, rho), f'expected `psi_matrix` to be close to the state |0><0| but received {rho}'


@pytest.mark.parametrize('sites', [[0, 1], [1, 0]])
def test_identity_2(sites):
    """Testing U rho U^-1 with U = I (x) I"""
    N = 2
    rho = identity(N)
    rho_true = np.copy(rho)
    rho_true = rho_true.reshape(2 ** N, 2 ** N)
    rho = two_qubit_gate(rho, np.kron(I, I).reshape([2] * 4), sites)
    rho = rho.reshape(2 ** N, 2 ** N)
    assert np.allclose(rho_true, rho), f'expected `psi_matrix` to be close to the state |0><0| but received {rho}'


def test_hadamard():
    """Testing U rho U^-1 with U = H (x) H """
    N = 2
    rho = identity(N)
    rho = single_qubit_gate(rho, H, 0)
    rho = single_qubit_gate(rho, H, 1)
    rho = rho.reshape(2 ** N, 2 ** N)
    rho_true = np.array([[0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j]])
    rho_true = np.outer(rho_true, rho_true.conj())
    assert np.allclose(rho_true, rho), f"expected `psi_true` to be equal to |x><x|, but obtained {rho}"


def test_paulix():
    """Testing U rho U^-1 with U = X (x) X """
    N = 2
    rho = identity(N)
    rho = single_qubit_gate(rho, X, 0)
    rho = single_qubit_gate(rho, X, 1)
    rho = rho.reshape(2 ** N, 2 ** N)
    rho_true = np.zeros((2 ** N, 2 ** N), complex)
    rho_true[-1, -1] = 1.0
    assert np.allclose(rho_true, rho), f"expected `psi_true` to be equal to |1><1|, but obtained {rho}"


@pytest.mark.parametrize('sites', [[0, 1], [1, 0]])
def test_bellstate_2(sites):
    """Creating 2 qubit Bell+ state"""
    N = 2
    rho = identity(N)
    rho = single_qubit_gate(rho, H, sites[0])
    rho = two_qubit_gate(rho, CNOT, sites)
    rho = rho.reshape(2 ** N, 2 ** N)
    rho_true = np.array([[1, 0, 0, 1]], dtype=complex) / np.sqrt(2)
    rho_true = np.outer(rho_true, rho_true.conj())
    assert np.allclose(rho_true, rho), f"expected `psi_true` to be equal to |bell+><bell+|, but obtained {rho}"


@pytest.mark.parametrize('sites', [[2, 1], [1, 2]])
def test_bellstate_3(sites):
    """Creating 2 qubit Bell+ state tensored with |0>"""
    N = 3
    rho = identity(N)
    rho = single_qubit_gate(rho, H, sites[0])
    rho = two_qubit_gate(rho, CNOT, sites)
    rho = rho.reshape(2 ** N, 2 ** N)
    rho_true = np.kron(np.array([[1, 0]]), np.array([[1, 0, 0, 1]], dtype=complex) / np.sqrt(2))
    rho_true = np.outer(rho_true, rho_true.conj())
    assert np.allclose(rho_true, rho), f"expected `psi_true` to be equal to |0>|bell+><bell+|<0|, but obtained {rho}"
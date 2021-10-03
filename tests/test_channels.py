import numpy as np

from glowjaw.channels import kraus_dep, single_qubit_channel
from glowjaw.ops import identity, to_matrix

import pytest

I = np.eye(2, 2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CNOT = CNOT.reshape([2] * 4)


def test_effect_to_full_mixed_single():
    """Testing if we get the mixed state with a fully depolarizing channel on a single qubit"""
    N = 1
    rho = identity(N)
    rho = single_qubit_channel(rho, kraus_dep(1.0), 0)
    assert np.allclose(rho, np.eye(2, 2, dtype=complex) * 0.5), 'Expected `rho` to be close to the fully mixed state.'


@pytest.mark.parametrize('N', [2, 3, 4, 5, 6])
def test_effect_to_full_mixed_N(N: int):
    """Testing if we get the mixed state with a fully depolarizing channel on N qubits"""
    rho = identity(N)
    for i in range(N):
        rho = single_qubit_channel(rho, kraus_dep(1.0), i)
    assert np.allclose(to_matrix(rho), np.eye(2 ** N, 2 ** N, dtype=complex) / (
                2 ** N)), 'Expected `rho` to be close to the fully mixed state.'

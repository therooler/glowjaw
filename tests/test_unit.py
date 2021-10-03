import numpy as np

from typing import List, Tuple, Union

from glowjaw.ops import identity, to_matrix,H,CNOT, to_tensor
from glowjaw.gates import single_qubit_gate, two_qubit_gate
from glowjaw.observables import single_qubit_observable, two_qubit_observable

import pytest
# Identity states

@pytest.mark.parametrize('N', [1.0, 'string', np.array([1])])
def test_identity(N: int):
    """Testing identity generation"""
    with pytest.raises(AssertionError, match='`N` must be an integer, received type'):
        identity(N)
    identity(4)

# Rho reshaping
@pytest.mark.parametrize('dims', [(2,3,2,3), [4,4], [10,]])
def test_reshape_to_matrix( dims: Union[List, Tuple[int, ...]]):
    """Testing reshaping to matrix"""
    rho = np.zeros(dims)
    with pytest.raises(AssertionError, match='Expected `rho` to be of shape '):
        to_matrix(rho)

@pytest.mark.parametrize('dims', [(2,2,2,2), [5,5], [10,]])
def test_reshape_to_matrix( dims: Union[List, Tuple[int, ...]]):
    """Testing reshaping to tensor"""
    rho = np.zeros(dims)
    with pytest.raises(AssertionError, match='Expected `rho` to be of shape '):
        to_tensor(rho)

# Single qubit gates

@pytest.mark.parametrize('state_input', [0, 1.0, 'sdf', (1, 0), [2, 2]])
def test_single_qubit_gate_state_input(state_input: np.ndarray):
    """Testing single qubit state input"""
    with pytest.raises(AssertionError, match='`state` must be a numpy array'):
        single_qubit_gate(state_input, H, 0)


@pytest.mark.parametrize('type', [int, float, np.float, np.complex64])
def test_single_qubit_gate_state_dtype(type):
    """Testing single qubit state dtype"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`state` must be a complex128 numpy array'):
        single_qubit_gate(rho.astype(type), H, 0)

    single_qubit_gate(rho.astype(complex), H, 0)


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 2, 1), (10, 1)])
def test_single_qubit_gate_state_shapes(shape):
    """Testing single qubit state shape"""
    rho = np.zeros(shape, dtype=complex)
    with pytest.raises(AssertionError, match='`state` must be of shape '):
        single_qubit_gate(rho, H, 0)


@pytest.mark.parametrize('gate_input', [0, 1.0, 'sdf', (1, 0), [2, 2]])
def test_single_qubit_gate_gate_input(gate_input: np.ndarray):
    """Testing single qubit gate input"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`gate` must be a numpy array'):
        single_qubit_gate(rho, gate_input, 0)


@pytest.mark.parametrize('type', [int, float, np.float, np.complex64])
def test_single_qubit_gate_gate_dtype(type):
    """Testing single qubit gate dtype"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`gate` must be a complex128 numpy array'):
        single_qubit_gate(rho, H.astype(type), 0)


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 2, 1), (10, 1)])
def test_single_qubit_gate_gate_shapes(shape):
    """Testing single qubit state shape"""
    N = 4
    rho = identity(N)
    H = np.zeros(shape, dtype=complex)
    with pytest.raises(AssertionError, match='`gate` must be of shape '):
        single_qubit_gate(rho, H, 0)


@pytest.mark.parametrize('gate_input', [np.random.rand(2, 2).astype(complex),
                                        np.ones((2, 2), dtype=complex)])
def test_single_qubit_gate_gate_unitary(gate_input):
    """Testing single qubit gate unitary"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`gate` must be unitary,'):
        single_qubit_gate(rho, gate_input, 0)


@pytest.mark.parametrize('loc_input', [1.0, [1, ], 'sdf', (1, 0), [2, 2]])
def test_single_qubit_gate_loc_input(loc_input: int):
    """Testing single qubit loc input"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`loc` must be an integer'):
        single_qubit_gate(rho, H, loc_input)


@pytest.mark.parametrize('loc_range', [-1, 5, 4])
def test_single_qubit_gate_loc_range(loc_range: int):
    """Testing single qubit loc range"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`loc` must be in range'):
        single_qubit_gate(rho, H, loc_range)


# 2 qubit gates

@pytest.mark.parametrize('state_input', [0, 1.0, 'sdf', (1, 0), [2, 2]])
def test_two_qubit_gate_state_input(state_input: np.ndarray):
    """Testing 2 qubit state input"""
    with pytest.raises(AssertionError, match='`state` must be a numpy array'):
        two_qubit_gate(state_input, H, (0, 1))


@pytest.mark.parametrize('type', [int, float, np.float, np.complex64])
def test_two_qubit_gate_state_dtype(type):
    """Testing 2 qubit state dtype"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`state` must be a complex128 numpy array'):
        two_qubit_gate(rho.astype(type), CNOT, (0, 1))


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 2, 1), (10, 1)])
def test_two_qubit_gate_state_shapes(shape):
    """Testing 2 qubit state shape"""
    rho = np.zeros(shape, dtype=complex)
    with pytest.raises(AssertionError, match='`state` must be of shape '):
        two_qubit_gate(rho, CNOT, (0, 1))


@pytest.mark.parametrize('gate_input', [0, 1.0, 'sdf', (1, 0), [2, 2]])
def test_two_qubit_gate_gate_input(gate_input: np.ndarray):
    """Testing 2 qubit gate input"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`gate` must be a numpy array'):
        two_qubit_gate(rho, gate_input, (0, 1))


@pytest.mark.parametrize('type', [int, float, np.float, np.complex64])
def test_two_qubit_gate_gate_dtype(type):
    """Testing 2 qubit gate dtype"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`gate` must be a complex128 numpy array'):
        two_qubit_gate(rho, CNOT.astype(type), (0, 1))


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 2, 1), (10, 1)])
def test_two_qubit_gate_gate_shapes(shape):
    """Testing 2 qubit state shape"""
    N = 4
    rho = identity(N)
    H = np.zeros(shape, dtype=complex)
    with pytest.raises(AssertionError, match='`gate` must be of shape '):
        two_qubit_gate(rho, H, (0, 1))


@pytest.mark.parametrize('gate_input', [np.random.rand(2, 2, 2, 2).astype(complex),
                                        np.ones((2, 2, 2, 2), dtype=complex)])
def test_two_qubit_gate_gate_unitary(gate_input):
    """Testing 2 qubit gate unitary"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`gate` must be unitary,'):
        two_qubit_gate(rho, gate_input, (0, 1))


@pytest.mark.parametrize('loc_input', [1.0, 1, 'sdf'])
def test_two_qubit_gate_loc_input(loc_input: Tuple[int, ...]):
    """Testing 2 qubit locs iterable"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`loc` must be iterable'):
        two_qubit_gate(rho, CNOT, loc_input)


@pytest.mark.parametrize('loc_input', [[1.0, 2.0], ('str', 'bla'), np.ones(2)])
def test_two_qubit_gate_loc_input(loc_input: Tuple[int, ...]):
    """Testing 2 qubit locs input"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`locs` must be an iterable of integers'):
        two_qubit_gate(rho, CNOT, loc_input)

    two_qubit_gate(rho, CNOT, np.array([1, 0]))


@pytest.mark.parametrize('loc_range', [(-1, 0), (5, 0), (4, 0)])
def test_two_qubit_gate_loc_range(loc_range: Tuple[int, ...]):
    """Testing 2 qubit locs range"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`loc`s must be in range'):
        two_qubit_gate(rho, CNOT, loc_range)


def test_two_qubit_gate_loc_unique():
    """Testing 2 qubit locs unique"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`locs` must contain unique integers'):
        two_qubit_gate(rho, CNOT, [1, 1])


# Single qubit observables

@pytest.mark.parametrize('state_input', [0, 1.0, 'sdf', (1, 0), [2, 2]])
def test_single_qubit_observable_state_input(state_input: np.ndarray):
    """Testing single qubit observable state input"""
    with pytest.raises(AssertionError, match='`state` must be a numpy array'):
        single_qubit_observable(state_input, H, 0)


@pytest.mark.parametrize('type', [int, float, np.float, np.complex64])
def test_single_qubit_observable_state_dtype(type):
    """Testing single qubit observable state dtype"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`state` must be a complex128 numpy array'):
        single_qubit_observable(rho.astype(type), H, 0)

    single_qubit_observable(rho.astype(complex), H, 0)


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 2, 1), (10, 1)])
def test_single_qubit_observable_state_shapes(shape):
    """Testing single qubit observable state shape"""
    rho = np.zeros(shape, dtype=complex)
    with pytest.raises(AssertionError, match='`state` must be of shape '):
        single_qubit_observable(rho, H, 0)


@pytest.mark.parametrize('gate_input', [0, 1.0, 'sdf', (1, 0), [2, 2]])
def test_single_qubit_observable_input(gate_input: np.ndarray):
    """Testing single qubit observable input"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`observable` must be a numpy array'):
        single_qubit_observable(rho, gate_input, 0)


@pytest.mark.parametrize('type', [int, float, np.float, np.complex64])
def test_single_qubit_observable_dtype(type):
    """Testing single qubit observable dtype"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`observable` must be a complex128 numpy array'):
        single_qubit_observable(rho, H.astype(type), 0)


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 2, 1), (10, 1)])
def test_single_qubit_observable_shapes(shape):
    """Testing single qubit observable state shape"""
    N = 4
    rho = identity(N)
    H = np.zeros(shape, dtype=complex)
    with pytest.raises(AssertionError, match='`observable` must be of shape '):
        single_qubit_observable(rho, H, 0)


@pytest.mark.parametrize('gate_input', [np.random.rand(2, 2).astype(complex),
                                        np.tril(np.ones((2, 2), dtype=complex))])
def test_single_qubit_observable_unitary(gate_input):
    """Testing single qubit observable hermitian"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`observable` must be hermitian,'):
        single_qubit_observable(rho, gate_input, 0)


@pytest.mark.parametrize('loc_input', [1.0, [1, ], 'sdf', (1, 0), [2, 2]])
def test_single_qubit_observable_loc_input(loc_input: int):
    """Testing single qubit observable loc input"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`loc` must be an integer'):
        single_qubit_observable(rho, H, loc_input)


@pytest.mark.parametrize('loc_range', [-1, 5, 4])
def test_single_qubit_observable_loc_range(loc_range: int):
    """Testing single qubit observable loc range"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`loc` must be in range'):
        single_qubit_observable(rho, H, loc_range)

# 2 qubit observables


@pytest.mark.parametrize('state_input', [0, 1.0, 'sdf', (1, 0), [2, 2]])
def test_two_qubit_observable_state_input(state_input: np.ndarray):
    """Testing 2 qubit state input"""
    with pytest.raises(AssertionError, match='`state` must be a numpy array'):
        two_qubit_observable(state_input, H, (0, 1))


@pytest.mark.parametrize('type', [int, float, np.float, np.complex64])
def test_two_qubit_observable_state_dtype(type):
    """Testing 2 qubit state dtype"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`state` must be a complex128 numpy array'):
        two_qubit_observable(rho.astype(type), CNOT, (0, 1))


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 2, 1), (10, 1)])
def test_two_qubit_observable_state_shapes(shape):
    """Testing 2 qubit state shape"""
    rho = np.zeros(shape, dtype=complex)
    with pytest.raises(AssertionError, match='`state` must be of shape '):
        two_qubit_observable(rho, CNOT, (0, 1))


@pytest.mark.parametrize('gate_input', [0, 1.0, 'sdf', (1, 0), [2, 2]])
def test_two_qubit_observable_input(gate_input: np.ndarray):
    """Testing 2 qubit observable input"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`observable` must be a numpy array'):
        two_qubit_observable(rho, gate_input, (0, 1))


@pytest.mark.parametrize('type', [int, float, np.float, np.complex64])
def test_two_qubit_observable_dtype(type):
    """Testing 2 qubit observable dtype"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`observable` must be a complex128 numpy array'):
        two_qubit_observable(rho, CNOT.astype(type), (0, 1))


@pytest.mark.parametrize('shape', [(2, 3), (2, 2, 2, 1), (10, 1)])
def test_two_qubit_observable_shapes(shape):
    """Testing 2 qubit observable state shape"""
    N = 4
    rho = identity(N)
    H = np.zeros(shape, dtype=complex)
    with pytest.raises(AssertionError, match='`observable` must be of shape '):
        two_qubit_observable(rho, H, (0, 1))


@pytest.mark.parametrize('gate_input', [np.random.rand(2, 2, 2, 2).astype(complex),
                                        np.tril(np.ones((2, 2, 2, 2), dtype=complex))])
def test_two_qubit_observable_unitary(gate_input):
    """Testing 2 qubit observable hermitian"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`observable` must be hermitian,'):
        two_qubit_observable(rho, gate_input, (0, 1))


@pytest.mark.parametrize('loc_input', [1.0, 1, 'sdf'])
def test_two_qubit_observable_loc_input(loc_input: Tuple[int, ...]):
    """Testing 2 qubit locs iterable"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`loc` must be iterable'):
        two_qubit_observable(rho, CNOT, loc_input)


@pytest.mark.parametrize('loc_input', [[1.0, 2.0], ('str', 'bla'), np.ones(2)])
def test_two_qubit_observable_loc_input(loc_input: Tuple[int, ...]):
    """Testing 2 qubit locs input"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`locs` must be an iterable of integers'):
        two_qubit_observable(rho, CNOT, loc_input)

    two_qubit_observable(rho, CNOT, np.array([1, 0]))


@pytest.mark.parametrize('loc_range', [(-1, 0), (5, 0), (4, 0)])
def test_two_qubit_observable_loc_range(loc_range: Tuple[int, ...]):
    """Testing 2 qubit locs range"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`loc`s must be in range'):
        two_qubit_observable(rho, CNOT, loc_range)


def test_two_qubit_observable_loc_unique():
    """Testing 2 qubit locs unique"""
    N = 4
    rho = identity(N)
    with pytest.raises(AssertionError, match='`locs` must contain unique integers'):
        two_qubit_observable(rho, CNOT, [1, 1])


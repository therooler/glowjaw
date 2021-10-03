import numpy as np
from typing import List, Union

# Pauli matrices
I = np.array([[1, 0], [0, 1]],dtype=complex)
H = np.array([[1, 1], [1, -1]],dtype=complex)/np.sqrt(2)
X = np.array([[0, 1], [1, 0]],dtype=complex)
Y = np.array([[0, -1j], [1j, 0]],dtype=complex)
Z = np.array([[1, 0], [0, -1]],dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CNOT = CNOT.reshape([2] * 4)
RX = lambda theta: np.cos(theta) * I - 1j * np.sin(theta) * X
RY = lambda theta: np.cos(theta) * I - 1j * np.sin(theta) * Y
RZ = lambda theta: np.cos(theta) * I - 1j * np.sin(theta) * Z


def assert_state(state: np.ndarray):
    assert isinstance(state,
                      np.ndarray), f'`state` must be a numpy array, received type {type(state)}'
    assert state.dtype == np.complex, f'`state` must be a complex128 numpy array, received type {state.dtype}'
    assert all(s == 2 for s in
               state.shape), f'`state` must be of shape [2,2,...,2], received shape {state.shape}'


def identity(nqubits: int) -> np.ndarray:
    r"""Create identity density matrix :math:`\rho=|0 \rangle \langle 0 |`.

    Args:
       nqubits: `int` specifying the number of qubits of the state.

    Returns:
        Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2].
    """
    assert np.issubdtype(type(nqubits),
                         int), f'`N` must be an integer, received type {type(nqubits)}'
    assert nqubits > 0, f'`N` must be larger than 0, received N = {nqubits}'
    state = np.zeros((2 ** nqubits, 2 ** nqubits), complex)
    state[0, 0] = 1.0
    return to_tensor(state)



def partial_trace(state: np.ndarray, dims_traced_out: List[int]) -> np.ndarray:
    r"""Return the partial trace of a density matrix :math:`\text{Tr}\left(\rho\right)`

    Args:
        state: Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2] or [2**N, 2**N].
        dims_traced_out: List of integers specifying which dimensions to trace out.
    Returns:
        Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2] or [2**N, 2**N].
    """
    assert_state(state)
    assert len(set(dims_traced_out)) == len(
        dims_traced_out), f'`dims_traced_out` must contain unique integers, receveived {dims_traced_out}'
    if len(state.shape) == 2:
        state = to_tensor(state)
    assert max(dims_traced_out) < len(
        state.shape) // 2, f'Max dim in `dims_traced_out` must be smaller than N = {len(state.shape) // 2}, ' \
                           f'received max(dims_traced_out) = {max(dims_traced_out)}'

    dims = state.shape
    nspins = len(dims) // 2
    idx_rho = list(range(2 * nspins))
    for dim in dims_traced_out:
        idx_rho[dim + nspins] = idx_rho[dim]
    return np.einsum(state, idx_rho)


def single_reprepare(state: np.ndarray, rep_state: np.ndarray, loc: int) -> np.ndarray:
    assert_state(state)
    assert_state(rep_state)
    dim_orgin = list(state.shape)
    nspins = len(dim_orgin) // 2
    state = partial_trace(state, [loc, ])
    print(trace(state))
    state = to_matrix(state)
    state = np.kron(rep_state, state, )
    state = state.reshape(dim_orgin)
    permutation = list(reversed(range(2 * nspins)))
    for p in range(nspins - loc - 1):
        permutation.insert(0, permutation.pop(-1))
        permutation.insert(0, permutation.pop(-1))
    return np.transpose(state, axes=permutation)


def trace(state: np.ndarray) -> Union[float, np.float]:
    r"""Return the trace of a density matrix :math:`\text{Tr}\left(\rho\right)`

    Args:
        state: Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2] or [2**N, 2**N].

    Returns:
        Real `float` corresponding to the trace of the density matrix.
    """
    if len(state.shape) != 2:
        state = to_matrix(state)

    return np.trace(state).real


def to_matrix(state):
    r"""Reshape the density matrix to a :math:`2^N \times 2^N` matrix.

    Args:
        state: Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2].

    Returns:
        Complex `np.ndarray` of shape (2**nqubits x 2**nqubits).
    """
    assert all(s == 2 for s in
               state.shape), f'Expected `rho` to be of shape [2,2,...,2], received shape {state.shape}'
    if state.shape == (2, 2):
        return state
    nqubits = len(state.shape) // 2
    return state.reshape(2 ** nqubits, 2 ** nqubits)


def to_tensor(state):
    r"""Reshape the density matrix to a tensor where all dimensions have shape 2.

    .. math::

        \rho_{ij} \to \rho_{i_1, i_2, \ldots, i_N, i_1', i_2', \ldots, i_N'}

    This shape makes it easier to apply local operators to the state.

    Args:
        state: Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2].

    Returns:
        Complex `np.ndarray` of shape (2**nqubits x 2**nqubits).
    """
    assert (len(state.shape) == 2) & (int(np.log2(state.shape[0])) == np.log2(
        state.shape[
            0])), f'Expected `rho` to be of shape [2**nqubits, 2**nqubits], received shape {state.shape}'
    nqubits = int(np.log2(state.shape[0]))
    return state.reshape([2] * (2 * nqubits))
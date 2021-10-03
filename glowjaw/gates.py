import numpy as np
from typing import List, Tuple, Union
from glowjaw.ops import assert_state



def single_qubit_gate(state: np.ndarray, gate: np.ndarray, loc: int) -> np.ndarray:
    r"""Apply a single single qubit unitary :math:`U` to the density matrix :math:`\rho` as

    .. math::

        \rho' = \quad U \rho U^\dag

    where :math:`U` is a unitary :math:`2\times 2` matrix.

    Args:
        state: Complex `np.ndarray` of (2**nqubits x 2**nqubits), with shape [2,2,...,2].
        gate: Complex `np.ndarray` unitary matrix of shape [2,2].
        loc: `int` specifying the qubit to which to apply the unitary.

    Returns:
        Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2].
    """
    assert np.issubdtype(type(loc), int), f'`loc` must be an integer, received loc = {type(loc)}'
    assert_state(state)
    assert isinstance(gate, np.ndarray), f'`gate` must be a numpy array, received type {type(gate)}'
    assert gate.dtype == np.complex, f'`gate` must be a complex128 numpy array, received type {gate.dtype}'
    assert gate.shape == (2, 2), f'`gate` must be of shape [2,2], received shape{gate.shape}'
    assert np.allclose(gate @ gate.conj().T,
                       np.eye(2, 2, dtype=complex)), f'`gate` must be unitary, but U @ U^dag != I'
    assert loc in range(
        len(state.shape) // 2), f'`loc` must be in range({len(state.shape) // 2}), received loc ={loc}'
    nqubits = len(state.shape) // 2
    indices_gate = [0, 1]
    indices_rho = list(range(2, 2 * nqubits + 2))
    indices_rho[loc] = 1
    indices_out = list(range(2, 2 * nqubits + 2))
    indices_out[loc] = 0
    state = np.einsum(gate, indices_gate, state, indices_rho, indices_out)
    indices_gate = [0, 1]
    indices_rho = list(range(2, 2 * nqubits + 2))
    indices_rho[loc + nqubits] = 1
    indices_out = list(range(2, 2 * nqubits + 2))
    indices_out[loc + nqubits] = 0
    return np.einsum(gate.conjugate(), indices_gate, state, indices_rho, indices_out)


def two_qubit_gate(state: np.ndarray, gate: np.ndarray,
                          locs: Union[Tuple[int, ...], List[int], np.ndarray]) -> np.ndarray:
    r"""Apply a 2 qubit unitary :math:`U` to the density matrix :math:`\rho` as

        .. math::

            \rho' = \quad U \rho U^\dag

        where :math:`U` is a unitary :math:`4\times 4` matrix.

        Args:
            state: Complex `np.ndarray` of (2**nqubits x 2**nqubits), with shape [2,2,...,2].
            gate: Complex `np.ndarray` unitary matrix of shape [2,2,2,2].
            locs: Iterable of 2 `int`s specifying the qubits to which to apply the unitary.

        Returns:
            Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2].
    """
    assert hasattr(locs, '__iter__'), '`locs` must be iterable.'
    assert all(np.issubdtype(type(l), int) for l in
               locs), f'`locs` must be an iterable of integers, received locs = {locs}'
    assert len(set(locs)) == len(
        locs), f'`locs` must contain unique integers, but received locs = {locs}'
    assert_state(state)

    assert isinstance(gate, np.ndarray), f'`gate` must be a numpy array, received type {type(gate)}'
    assert gate.dtype == np.complex, f'`gate` must be a complex128 numpy array, received type {gate.dtype}'
    assert gate.shape == (
    2, 2, 2, 2), f'`gate` must be of shape [2,2,2,2], received shape {gate.shape}'
    assert np.allclose(gate.reshape(4, 4) @ gate.reshape(4, 4).conj().T,
                       np.eye(4, 4, dtype=complex)), f'`gate` must be unitary, but U @ U^dag != I'
    assert all(l in range(len(state.shape) // 2) for l in
               locs), f'all `loc`s must be in range({len(state.shape) // 2}), received locs = {locs}'
    nqubits = len(state.shape) // 2
    indices_gate = [0, 1, 2, 3]
    indices_rho = list(range(4, 2 * nqubits + 4))
    indices_rho[locs[0]] = 2
    indices_rho[locs[1]] = 3
    indices_out = list(range(4, 2 * nqubits + 4))
    indices_out[locs[0]] = 0
    indices_out[locs[1]] = 1

    state = np.einsum(gate, indices_gate, state, indices_rho, indices_out)
    indices_gate = [0, 1, 2, 3]
    indices_rho = list(range(4, 2 * nqubits + 4))
    indices_rho[locs[0] + nqubits] = 2
    indices_rho[locs[1] + nqubits] = 3
    indices_out = list(range(4, 2 * nqubits + 4))
    indices_out[locs[0] + nqubits] = 0
    indices_out[locs[1] + nqubits] = 1

    return np.einsum(gate.conjugate(), indices_gate, state, indices_rho, indices_out)
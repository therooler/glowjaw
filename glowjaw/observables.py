import numpy as np
from typing import List, Tuple, Union
from glowjaw.ops import assert_state

def single_qubit_observable(state: np.ndarray, observable: np.ndarray,
                                     loc: int) -> np.ndarray:
    r"""Measure a single qubit observable :math:`\langle O \rangle` on the density matrix :math:`\rho` as

        .. math::

            \langle O \rangle = \text{Tr}\left(\rho O \right)

        where :math:`O` is a Hermitian :math:`2\times 2` matrix.

        Args:
            state: Complex `np.ndarray` of (2**nqubits x 2**nqubits), with shape [2,2,...,2].
            observable: Complex `np.ndarray` Hermitian matrix of shape [2,2].
            loc: `int` specifying the qubit to be measured.

        Returns:
            Real `float` corresponding to the desired observable.
    """
    assert np.issubdtype(type(loc), int), f'`loc` must be an integer, received loc = {type(loc)}'
    assert_state(state)
    assert isinstance(observable,
                      np.ndarray), f'`observable` must be a numpy array, received type {type(observable)}'
    assert observable.dtype == np.complex, f'`observable` must be a complex128 numpy array, received type {observable.dtype}'
    assert observable.shape == (
    2, 2), f'`observable` must be of shape [2,2], received shape{observable.shape}'
    assert np.allclose(observable,
                       observable.conj().T), f'`observable` must be hermitian, but O != O^dag'
    assert loc in range(
        len(state.shape) // 2), f'`loc` must be in range({len(state.shape) // 2}), received loc ={loc}'
    nqubits = len(state.shape) // 2
    indices_gate = [0, 1]
    indices_rho = list(range(2, 2 * nqubits + 2))
    indices_rho[loc] = 1
    indices_out = list(range(2, 2 * nqubits + 2))
    indices_out[loc] = 0
    state = np.einsum(observable, indices_gate, state, indices_rho, indices_out)
    indices_rho = list(range(nqubits)) * 2
    return np.einsum(state, indices_rho).real


def two_qubit_observable(state: np.ndarray, observable: np.ndarray,
                                locs: Union[Tuple[int, ...], List[int], np.ndarray]) -> np.ndarray:
    r"""Measure a 2 qubit observable :math:`\langle O \rangle` on the density matrix :math:`\rho` as

            .. math::

                \langle O \rangle = \text{Tr}\left(\rho O \right)

            where :math:`O` is a Hermitian :math:`4\times 4` matrix.

            Args:
                state: Complex `np.ndarray` of (2**nqubits x 2**nqubits), with shape [2,2,...,2].
                observable: Complex `np.ndarray` Hermitian matrix of shape [2,2,2,2].
                loc: Iterable of two `int`s specifying the qubits to be measured.

            Returns:
                Real `float` corresponding to the desired observable.
    """
    assert hasattr(locs, '__iter__'), '`locs` must be iterable.'
    assert all(np.issubdtype(type(l), int) for l in
               locs), f'`locs` must be an iterable of integers, received locs = {locs}'
    assert len(set(locs)) == len(
        locs), f'`locs` must contain unique integers, but received locs = {locs}'
    assert_state(state)
    assert isinstance(observable,
                      np.ndarray), f'`observable` must be a numpy array, received type {type(observable)}'
    assert observable.dtype == np.complex, f'`observable` must be a complex128 numpy array,' \
                                           f' received type {observable.dtype}'
    assert observable.shape == (2, 2, 2, 2), \
        f'`observable` must be of shape [2,2,2,2], received shape {observable.shape}'
    assert np.allclose(observable.reshape(4, 4), observable.reshape(4, 4).conj().T,
                       np.eye(4, 4,
                              dtype=complex)), f'`observable` must be hermitian, but O != O^dag'
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
    state = np.einsum(observable, indices_gate, state, indices_rho, indices_out)
    indices_rho = list(range(nqubits)) * 2
    return np.einsum(state, indices_rho).real

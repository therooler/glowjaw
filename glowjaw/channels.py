import numpy as np
from typing import List, Tuple, Union
from glowjaw.ops import assert_state, trace, I,X,Y,Z
import warnings
from functools import reduce


def kraus_dep(noise):
    """
    Defines the Kraus operators of the depolarising channel of a given noise
    degree.
    """
    Dep = np.zeros((4, 2, 2), complex)
    Dep[0] = np.sqrt((4 - 3 * noise) / 4) * I
    Dep[1] = np.sqrt(noise / 4) * X
    Dep[2] = np.sqrt(noise / 4) * Y
    Dep[3] = np.sqrt(noise / 4) * Z
    return Dep


def single_qubit_projector(state: np.ndarray, projector: np.ndarray,
                                    loc: int) -> np.ndarray:
    r"""Apply a single qubit projector :math:`\Pi` on the density matrix :math:`\rho` as

        .. math::

            \rho' = \rho \Pi/\text{Tr}\left(\rho \Pi \right)

        where :math:`\Pi` is an idempotent Hermitian :math:`2\times 2` matrix. If `projector` is orthogonal to the state,
        we throw a warning.

        Args:
            state: Complex `np.ndarray` of (2**nqubits x 2**nqubits), with shape [2,2,...,2].
            observable: Complex `np.ndarray` idempotent projection matrix of shape [2,2].
            loc: `int` specifying the qubit to be measured.

        Returns:
            Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2].
    """
    assert np.issubdtype(type(loc), int), f'`loc` must be an integer, received loc = {type(loc)}'
    assert_state(state)
    assert isinstance(projector,
                      np.ndarray), f'`projector` must be a numpy array, received type {type(projector)}'
    assert projector.dtype == np.complex, f'`projector` must be a complex128 numpy array, received type {projector.dtype}'
    assert projector.shape == (
    2, 2), f'`projector` must be of shape [2,2], received shape{projector.shape}'
    assert np.allclose(projector,
                       projector.conj().T), f'`projector` must be hermitian, but O != O^dag'
    assert np.allclose(projector,
                       projector @ projector), f'`projector` must be idempotent, but P != P^2'
    assert loc in range(
        len(state.shape) // 2), f'`loc` must be in range({len(state.shape) // 2}), received loc ={loc}'
    nqubits = len(state.shape) // 2
    indices_gate = [0, 1]
    indices_rho = list(range(2, 2 * nqubits + 2))
    indices_rho[loc] = 1
    indices_out = list(range(2, 2 * nqubits + 2))
    indices_out[loc] = 0
    state = np.einsum(projector, indices_gate, state, indices_rho, indices_out)
    indices_gate = [0, 1]
    indices_rho = list(range(2, 2 * nqubits + 2))
    indices_rho[loc + nqubits] = 1
    indices_out = list(range(2, 2 * nqubits + 2))
    indices_out[loc + nqubits] = 0
    state = np.einsum(projector, indices_gate, state, indices_rho, indices_out)
    norm = trace(state)
    if np.isclose(norm, 0.):
        warnings.warn(
            RuntimeWarning('Divide by zero encountered because projector is orthogonal to `rho`.'))
    return state / norm


def single_qubit_projector_probability(state: np.ndarray, projector: np.ndarray,
                                                loc: int) -> Union[
    float, np.float]:
    r"""Get the probability :math:`p_\Pi` of finding the state

        .. math::

            \rho' = \rho \Pi/\text{Tr}\left(\rho \Pi \right)

        where :math:`\Pi` is an idempotent Hermitian :math:`2\times 2` matrix given by

        .. math::

            p_\Pi = \text{Tr}\left(\rho \Pi \right)

        Args:
            state: Complex `np.ndarray` of (2**nqubits x 2**nqubits), with shape [2,2,...,2].
            observable: Complex `np.ndarray` idempotent projection matrix of shape [2,2].
            loc: `int` specifying the qubit to be measured.

        Returns:
            Real `float` corresponding to the probability of observing the projected state.
    """
    assert np.issubdtype(type(loc), int), f'`loc` must be an integer, received loc = {type(loc)}'
    assert_state(state)
    assert isinstance(projector,
                      np.ndarray), f'`projector` must be a numpy array, received type {type(projector)}'
    assert projector.dtype == np.complex, f'`projector` must be a complex128 numpy array, received type {projector.dtype}'
    assert projector.shape == (
    2, 2), f'`projector` must be of shape [2,2], received shape{projector.shape}'
    assert np.allclose(projector,
                       projector.conj().T), f'`projector` must be hermitian, but O != O^dag'
    assert np.allclose(projector,
                       projector @ projector), f'`projector` must be idempotent, but P != P^2'
    assert loc in range(
        len(state.shape) // 2), f'`loc` must be in range({len(state.shape) // 2}), received loc ={loc}'
    nqubits = len(state.shape) // 2
    indices_gate = [0, 1]
    indices_rho = list(range(2, 2 * nqubits + 2))
    indices_rho[loc] = 1
    indices_out = list(range(2, 2 * nqubits + 2))
    indices_out[loc] = 0
    state = np.einsum(projector, indices_gate, state, indices_rho, indices_out)

    return trace(state)


def single_qubit_effect(state: np.ndarray, effect: np.ndarray, loc: int) -> np.ndarray:
    r"""Apply a single qubit single_qubit_effect :math:`M_a` on the density matrix :math:`\rho` as

        .. math::

            \rho' = \rho M_a/\text{Tr}\left(\rho M_a \right)

        where :math:`M_a` is an :math:`2\times 2` matrix. If `effect` is orthogonal to the state,
        we throw an warning.

        Args:
            state: Complex `np.ndarray` of (2**nqubits x 2**nqubits), with shape [2,2,...,2].
            observable: Complex `np.ndarray` idempotent projection matrix of shape [2,2].
            loc: `int` specifying the qubit to be measured.

        Returns:
            Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2].
    """
    assert np.issubdtype(type(loc), int), f'`loc` must be an integer, received loc = {type(loc)}'
    assert_state(state)
    assert isinstance(effect,
                      np.ndarray), f'`effect` must be a numpy array, received type {type(effect)}'
    assert effect.dtype == np.complex, f'`effect` must be a complex128 numpy array, received type {effect.dtype}'
    assert effect.shape == (2, 2), f'`effect` must be of shape [2,2], received shape{effect.shape}'
    assert loc in range(
        len(state.shape) // 2), f'`loc` must be in range({len(state.shape) // 2}), received loc ={loc}'
    nqubits = len(state.shape) // 2
    indices_gate = [0, 1]
    indices_rho = list(range(2, 2 * nqubits + 2))
    indices_rho[loc] = 1
    indices_out = list(range(2, 2 * nqubits + 2))
    indices_out[loc] = 0
    state = np.einsum(effect, indices_gate, state, indices_rho, indices_out)
    indices_gate = [0, 1]
    indices_rho = list(range(2, 2 * nqubits + 2))
    indices_rho[loc + nqubits] = 1
    indices_out = list(range(2, 2 * nqubits + 2))
    indices_out[loc + nqubits] = 0
    state = np.einsum(effect.conj(), indices_gate, state, indices_rho, indices_out)
    norm = trace(state)
    # print(norm, 'norm in eff')
    if np.isclose(norm, 0.):
        print(norm)
        warnings.warn(
            RuntimeWarning('Divide by zero encountered because projector is orthogonal to `rho`.'))
    return state / norm


def single_qubit_effect_probability(state: np.ndarray, effect: np.ndarray, loc: int) -> \
Union[
    float, np.float]:
    r"""Get the probability :math:`p_{M_a}` of finding the state

        .. math::

            \rho' = \rho M_a/\text{Tr}\left(\rho M_a \right)

        where :math:`M` is a :math:`2\times 2` effect matrix.

        Args:
            state: Complex `np.ndarray` of (2**nqubits x 2**nqubits), with shape [2,2,...,2].
            observable: Complex `np.ndarray` effect matrix of shape [2,2].
            loc: `int` specifying the qubit to be measured.

        Returns:
            Real `float` corresponding to the probability of observing the projected state.
    """
    assert np.issubdtype(type(loc), int), f'`loc` must be an integer, received loc = {type(loc)}'
    assert_state(state)
    assert isinstance(effect,
                      np.ndarray), f'`effect` must be a numpy array, received type {type(effect)}'
    assert effect.dtype == np.complex, f'`effect` must be a complex128 numpy array, received type {effect.dtype}'
    assert effect.shape == (2, 2), f'`effect` must be of shape [2,2], received shape{effect.shape}'
    assert loc in range(
        len(state.shape) // 2), f'`loc` must be in range({len(state.shape) // 2}), received loc ={loc}'
    nqubits = len(state.shape) // 2
    indices_gate = [0, 1]
    indices_rho = list(range(2, 2 * nqubits + 2))
    indices_rho[loc] = 1
    indices_out = list(range(2, 2 * nqubits + 2))
    indices_out[loc] = 0
    state = np.einsum(effect, indices_gate, state, indices_rho, indices_out)
    indices_gate = [0, 1]
    indices_rho = list(range(2, 2 * nqubits + 2))
    indices_rho[loc + nqubits] = 1
    indices_out = list(range(2, 2 * nqubits + 2))
    indices_out[loc + nqubits] = 0
    state = np.einsum(effect.conj(), indices_gate, state, indices_rho, indices_out)
    return trace(state)


def two_qubit_effect_probability(state: np.ndarray, effect: np.ndarray,
                                        locs: Union[Tuple[int, ...], List[int], np.ndarray]) -> \
Union[
    float, np.float]:
    r"""Get the probability :math:`p_{M_a, M_b}` of finding the state

        .. math::

            \rho' = \rho M_a, M_b / \text{Tr}\left(\rho M_a, M_b \right)

        where :math:`M` is a :math:`4\times 4` effect matrix given by :math:`M_a \otimes M_b`.

        Args:
            state: Complex `np.ndarray` of (2**nqubits x 2**nqubits), with shape [2,2,...,2].
            observable: Complex `np.ndarray` effect matrix of shape [2,2,2,2].
            locs: Iterable of 2 `int`s specifying the qubits to which to apply the unitary.

        Returns:
            Real `float` corresponding to the probability of observing the projected state.
    """
    assert hasattr(locs, '__iter__'), '`locs` must be iterable.'
    assert all(np.issubdtype(type(l), int) for l in
               locs), f'`locs` must be an iterable of integers, received locs = {locs}'
    assert_state(state)
    assert isinstance(effect,
                      np.ndarray), f'`effect` must be a numpy array, received type {type(effect)}'
    assert effect.dtype == np.complex, f'`effect` must be a complex128 numpy array, received type {effect.dtype}'
    assert effect.shape == (
    2, 2, 2, 2), f'`effect` must be of shape [2,2,2,2], received shape{effect.shape}'

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
    state = np.einsum(effect, indices_gate, state, indices_rho, indices_out)
    return trace(state)


def two_qubit_effect(state: np.ndarray, effect: np.ndarray,
                            locs: Union[Tuple[int, ...], List[int], np.ndarray]) -> np.ndarray:
    r"""Apply a 2 qubit effect :math:`M_a \otimes M_b` on the density matrix :math:`\rho` as

    .. math::

        \rho' = \rho \otimes M_b/\text{Tr}\left(\rho M_a \otimes M_b \right)

    where :math:`M_a` is an :math:`2\times 2` matrix. If `effect` is orthogonal to the state,
    we throw an warning.

    Args:
        state: Complex `np.ndarray` of (2**nqubits x 2**nqubits), with shape [2,2,...,2].
        observable: Complex `np.ndarray` effect matrix of shape [2,2,2,2].
        locs: Iterable of 2 `int`s specifying the qubits to which to apply the unitary.

    Returns:
        Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2].
    """
    assert hasattr(locs, '__iter__'), '`locs` must be iterable.'
    assert all(np.issubdtype(type(l), int) for l in
               locs), f'`locs` must be an iterable of integers, received locs = {locs}'
    assert_state(state)
    assert isinstance(effect,
                      np.ndarray), f'`effect` must be a numpy array, received type {type(effect)}'
    assert effect.dtype == np.complex, f'`effect` must be a complex128 numpy array, received type {effect.dtype}'
    assert effect.shape == (
    2, 2, 2, 2), f'`effect` must be of shape [2,2], received shape{effect.shape}'

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
    state = np.einsum(effect, indices_gate, state, indices_rho, indices_out)
    indices_gate = [0, 1, 2, 3]
    indices_rho = list(range(4, 2 * nqubits + 4))
    indices_rho[locs[0] + nqubits] = 2
    indices_rho[locs[1] + nqubits] = 3
    indices_out = list(range(4, 2 * nqubits + 4))
    indices_out[locs[0] + nqubits] = 0
    indices_out[locs[1] + nqubits] = 1
    state = np.einsum(effect.conj(), indices_gate, state, indices_rho, indices_out)
    norm = trace(state)
    if np.isclose(norm, 0.):
        print(norm)
        warnings.warn(
            RuntimeWarning('Divide by zero encountered because projector is orthogonal to `rho`.'))
    return state / norm


def single_qubit_channel(state: np.ndarray, effect: np.ndarray, loc: int) -> np.ndarray:
    r"""Apply a single qubit kraus channel to the density matrix :math:`rho`

    .. math::

        \rho' = \sum_i K_i \rho K^\dag_i

    where

    .. math::

        \sum_i K_i K^\dag_i = I

    and :math:`K_i` is a :math:`2 \times 2` hermitian operator.

    Args:
        state: Complex `np.ndarray` of (2**nqubits x 2**nqubits), with shape [2,2,...,2].
        effect: Collection of K complex `np.ndarray` hermitian operators with shape [K,2,2].
        loc: `int` specifying the qubit onto which to apply the channel.

    Returns:
        Complex `np.ndarray` of (2**nqubits x 2**nqubits) elements, with shape [2,2,...,2].
    """
    assert np.issubdtype(type(loc), int), f'`loc` must be an integer, received loc = {type(loc)}'
    assert_state(state)
    assert isinstance(effect,
                      np.ndarray), f'`effect` must be a numpy array, received type {type(effect)}'
    assert effect.dtype == np.complex, f'`effect` must be a complex128 numpy array, received type {effect.dtype}'
    assert (len(effect.shape) == 3) & (
            effect.shape[1:] == (
    2, 2)), f'`effect` must be of shape [K,2,2], received shape{effect.shape}'
    assert all(
        np.allclose(K, K.conj().T) for K in effect), f'`effect` must be hermitian, but O != O^dag'
    assert np.allclose(reduce(np.add, [K @ K.conj().T for K in effect]),
                       np.eye(2, 2)), f'sum_i K_i @ K_i^dag must ' \
                                      f'give the identity, but received{np.sum(effect, axis=0)}'
    assert loc in range(
        len(state.shape) // 2), f'`loc` must be in range({len(state.shape) // 2}), received loc ={loc}'
    state_dep = np.zeros_like(state)
    for eff in effect:
        nqubits = len(state.shape) // 2
        indices_gate = [0, 1]
        indices_rho = list(range(2, 2 * nqubits + 2))
        indices_rho[loc] = 1
        indices_out = list(range(2, 2 * nqubits + 2))
        indices_out[loc] = 0
        state_intermediate = np.einsum(eff, indices_gate, state, indices_rho, indices_out)
        indices_gate = [0, 1]
        indices_rho = list(range(2, 2 * nqubits + 2))
        indices_rho[loc + nqubits] = 1
        indices_out = list(range(2, 2 * nqubits + 2))
        indices_out[loc + nqubits] = 0
        state_dep += np.einsum(eff.conj(), indices_gate, state_intermediate, indices_rho,
                               indices_out)
    return state_dep

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# State generators
# ---------------------------------------------------------------------------

def haar_random_state(n_qubits: int, rng=None) -> np.ndarray:
    """
    Haar-uniform pure state on a 2^n-dimensional Hilbert space.

    Samples a complex Gaussian vector and normalises it.  The resulting
    distribution is invariant under all unitaries -- the natural notion of
    'uniformly random' on a complex Hilbert space.
    """
    if rng is None:
        rng = np.random.default_rng()
    dim = 2 ** n_qubits
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return psi / np.linalg.norm(psi)


def computational_basis_state(n_qubits: int, index: int = 0) -> np.ndarray:
    """
    Computational basis state |index> as a state vector.
    Bit ordering is big-endian: index=0 -> |00...0>, index=1 -> |00...1>, etc.
    """
    psi = np.zeros(2 ** n_qubits, dtype=complex)
    psi[index] = 1.0
    return psi


def product_state(*states: np.ndarray) -> np.ndarray:
    """Tensor (Kronecker) product of single-qubit state vectors."""
    result = states[0].astype(complex)
    for s in states[1:]:
        result = np.kron(result, s.astype(complex))
    return result


def dm_from_ket(psi: np.ndarray) -> np.ndarray:
    """Pure-state density matrix  rho = |psi><psi|."""
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T


def dm_from_ensemble(states: List[np.ndarray], probs: List[float]) -> np.ndarray:
    """
    Mixed-state density matrix from a convex combination:
        rho = sum_i  p_i |psi_i><psi_i|
    Requires sum(probs) == 1.
    """
    assert abs(sum(probs) - 1) < 1e-8, "Probabilities must sum to 1."
    d = len(states[0])
    rho = np.zeros((d, d), dtype=complex)
    for psi, p in zip(states, probs):
        rho += p * dm_from_ket(psi)
    return rho


# ---------------------------------------------------------------------------
# Noise channels
# ---------------------------------------------------------------------------

def depolarising_noise(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Depolarising channel: rho -> (1-p)*rho + (p/d)*I
    Replaces the state with the maximally-mixed state with probability p.
    p=0 -> identity;  p=1 -> maximally mixed I/d.
    """
    d = rho.shape[0]
    return (1 - p) * rho + (p / d) * np.eye(d, dtype=complex)


def dephasing_noise(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Dephasing channel: suppresses off-diagonal coherences by factor (1-p).
        rho_ij -> (1-p)*rho_ij  for i!=j;  rho_ii unchanged.
    p=1 gives the fully classical diagonal state.
    """
    rho_n = rho.copy()
    rho_n *= (1.0 - p * (1.0 - np.eye(rho.shape[0])))
    return rho_n


def amplitude_damping_noise(rho: np.ndarray, gamma: float) -> np.ndarray:
    """
    Amplitude-damping channel: models spontaneous decay |1> -> |0> or
    photon loss.  Kraus operators K0, K1 applied independently per qubit
    (local / product noise model).

    K0 = [[1,0],[0,sqrt(1-gamma)]]  (no decay)
    K1 = [[0,sqrt(gamma)],[0,0]]    (decay: excitation lost)
    """
    from library import _embed_single  # local import to avoid circular dependency

    nq = int(round(np.log2(rho.shape[0])))
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    for q in range(nq):
        tmp = np.zeros_like(rho)
        for K in (K0, K1):
            Kf = _embed_single(K, q, nq)
            tmp += Kf @ rho @ Kf.conj().T
        rho = tmp
    return rho


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def generate_dataset(
    n_samples: int,
    n_qubits: int = 2,
    noise_prob: float = 0.0,
    noise_type: str = "depolarising",
    seed=None,
) -> List[Dict]:
    """
    Generate a reproducible dataset of quantum density matrices.

    Each sample starts as a Haar-random pure state; the chosen noise channel
    is applied if noise_prob > 0.

    Parameters
    ----------
    n_samples  : number of states to generate
    n_qubits   : Hilbert-space dimension = 2^n_qubits
    noise_prob : noise strength in [0,1];  0 -> pure-state dataset
    noise_type : 'depolarising' | 'dephasing' | 'amplitude_damping' | 'none'
    seed       : integer seed for full reproducibility

    Returns
    -------
    List of dicts with keys:
        psi, rho_pure, rho, n_qubits, purity, noise_type, noise_prob
    """
    rng, out = np.random.default_rng(seed), []
    for _ in range(n_samples):
        psi = haar_random_state(n_qubits, rng)
        rp = dm_from_ket(psi)
        if noise_prob > 0 and noise_type != "none":
            fn = {
                "depolarising": depolarising_noise,
                "dephasing": dephasing_noise,
                "amplitude_damping": amplitude_damping_noise,
            }.get(noise_type)
            if fn is None:
                raise ValueError(f"Unknown noise_type '{noise_type}'")
            rho = fn(rp, noise_prob)
        else:
            rho = rp.copy()
        out.append(
            {
                "psi": psi,
                "rho_pure": rp,
                "rho": rho,
                "n_qubits": n_qubits,
                "purity": float(np.real(np.trace(rho @ rho))),
                "noise_type": noise_type,
                "noise_prob": noise_prob,
            }
        )
    return out

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# Import state-generation primitives (no circular dependency)
from generate_state import (
    computational_basis_state,
    dm_from_ket,
)


# ---------------------------------------------------------------------------
# Single-qubit (2×2) optical components
# ---------------------------------------------------------------------------

def phase_shifter(phi: float) -> np.ndarray:
    """
    Single-mode phase shifter.
    PS(phi) = diag(1, e^{i*phi})
    """
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)


def half_wave_plate(theta: float) -> np.ndarray:
    """
    Half-wave plate with fast axis at angle theta.
    HWP(theta) = [[cos2t, sin2t], [sin2t, -cos2t]]
    """
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    return np.array([[c2, s2], [s2, -c2]], dtype=complex)


def quarter_wave_plate(theta: float) -> np.ndarray:
    """
    Quarter-wave plate with fast axis at angle theta.
    Introduces pi/2 retardation; converts linear <-> circular polarisation.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.exp(1j * np.pi / 4) * np.array(
        [[c ** 2 + 1j * s ** 2, (1 - 1j) * s * c],
         [(1 - 1j) * s * c,    s ** 2 + 1j * c ** 2]],
        dtype=complex,
    )


def beam_splitter_2mode(r=1 / np.sqrt(2), t=1j / np.sqrt(2)) -> np.ndarray:
    """
    2×2 beam-splitter matrix (one polarisation component).
    BS = [[t, r], [r, t]].  Default: symmetric lossless 50:50 splitter.
    """
    return np.array([[t, r], [r, t]], dtype=complex)


# ---------------------------------------------------------------------------
# Two-qubit (4×4) optical components
# ---------------------------------------------------------------------------

def polarising_beam_splitter() -> np.ndarray:
    """
    Polarising beam splitter (PBS) on two polarisation-encoded qubits.
    Basis: {|HH>, |HV>, |VH>, |VV>}.
    """
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )


def su2_gate(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Universal single-qubit SU(2) gate via ZYZ Euler decomposition:
        U(alpha,beta,gamma) = Rz(alpha) * Ry(beta) * Rz(gamma)
    """
    Rz = lambda a: np.array(
        [[np.exp(-1j * a / 2), 0], [0, np.exp(1j * a / 2)]], dtype=complex
    )
    Ry = lambda b: np.array(
        [[np.cos(b / 2), -np.sin(b / 2)], [np.sin(b / 2), np.cos(b / 2)]],
        dtype=complex,
    )
    return Rz(alpha) @ Ry(beta) @ Rz(gamma)


def cnot_gate() -> np.ndarray:
    """CNOT gate — flips target qubit iff control is |1>."""
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )


def cz_gate() -> np.ndarray:
    """Controlled-Z gate — applies a pi phase to |11> only."""
    return np.diag([1, 1, 1, -1]).astype(complex)


def swap_gate() -> np.ndarray:
    """SWAP gate — exchanges the states of two qubits: |ab> -> |ba>."""
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )


# ---------------------------------------------------------------------------
# SPDC sources
# ---------------------------------------------------------------------------

def spdc_bell_state(variant: str = "phi_plus") -> np.ndarray:
    """
    Two-qubit Bell state from one ideal SPDC photon-pair source.
    Encoding: |H> <-> |0>,  |V> <-> |1>.

    Variants: 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'
    """
    r2 = np.sqrt(2)
    table = {
        "phi_plus":  np.array([1, 0, 0, 1], dtype=complex) / r2,
        "phi_minus": np.array([1, 0, 0, -1], dtype=complex) / r2,
        "psi_plus":  np.array([0, 1, 1, 0], dtype=complex) / r2,
        "psi_minus": np.array([0, 1, -1, 0], dtype=complex) / r2,
    }
    if variant not in table:
        raise ValueError(f"Unknown variant '{variant}'. Options: {list(table)}")
    return table[variant]


def auto_n_spdc(n_qubits: int) -> int:
    """
    Number of SPDC sources to cover n_qubits.
    Uses ceil(n_qubits/2) sources, capped at 3.
    """
    return min(3, max(1, (n_qubits + 1) // 2))


def initial_state_from_spdc(n_qubits: int, n_spdc: int) -> np.ndarray:
    """
    Build the initial n_qubits-qubit state from multiple SPDC sources.
    """
    if not 1 <= n_spdc <= 3:
        raise ValueError(f"n_spdc must be 1-3, got {n_spdc}")
    if n_spdc * 2 > n_qubits:
        raise ValueError(
            f"{n_spdc} sources need {n_spdc * 2} qubits but n_qubits={n_qubits}"
        )
    bell = spdc_bell_state("phi_plus")
    psi = bell
    for _ in range(n_spdc - 1):
        psi = np.kron(psi, bell)
    rem = n_qubits - n_spdc * 2
    if rem > 0:
        psi = np.kron(psi, computational_basis_state(rem, 0))
    return psi


# ---------------------------------------------------------------------------
# Gate embedding helpers
# ---------------------------------------------------------------------------

def _embed_single(U: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Lift a 2×2 single-qubit unitary into the 2^n_qubits Hilbert space."""
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = U
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _embed_two(U4: np.ndarray, q1: int, q2: int, n_qubits: int) -> np.ndarray:
    """Lift a 4×4 two-qubit unitary into the 2^n_qubits Hilbert space."""
    dim = 2 ** n_qubits
    Uf = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            bi1 = (i >> (n_qubits - 1 - q1)) & 1
            bi2 = (i >> (n_qubits - 1 - q2)) & 1
            bj1 = (j >> (n_qubits - 1 - q1)) & 1
            bj2 = (j >> (n_qubits - 1 - q2)) & 1
            same = all(
                ((i >> (n_qubits - 1 - k)) & 1) == ((j >> (n_qubits - 1 - k)) & 1)
                for k in range(n_qubits)
                if k not in (q1, q2)
            )
            if same:
                Uf[i, j] = U4[bi1 * 2 + bi2, bj1 * 2 + bj2]
    return Uf


def _permute_qubits(rho: np.ndarray, perm: List[int], n_qubits: int) -> np.ndarray:
    """Permute the qubit ordering of a density matrix."""
    t = rho.reshape([2] * (2 * n_qubits))
    full_perm = list(perm) + [p + n_qubits for p in perm]
    return np.transpose(t, full_perm).reshape(2 ** n_qubits, 2 ** n_qubits)


# ---------------------------------------------------------------------------
# Gate descriptor and registry
# ---------------------------------------------------------------------------

@dataclass
class GateOp:
    """
    Descriptor for one gate in a photonic circuit.
    name   : key in GATE_REGISTRY
    qubits : qubit indices (length = gate arity)
    params : continuous parameters (angles in radians, phases, etc.)
    """
    name:   str
    qubits: List[int]
    params: List[float] = field(default_factory=list)

    def __repr__(self):
        ps = ", ".join(f"{p:.4f}" for p in self.params)
        return f"{self.name}(q={self.qubits}" + (f", p=[{ps}])" if self.params else ")")

    def copy(self):
        return GateOp(self.name, self.qubits.copy(), self.params.copy())


# Format: name -> (qubit_arity, [(param_lo, param_hi), ...])
GATE_REGISTRY: Dict[str, Tuple[int, List[Tuple[float, float]]]] = {
    "PS":   (1, [(0.0, 2 * np.pi)]),
    "HWP":  (1, [(0.0, np.pi / 2)]),
    "QWP":  (1, [(0.0, np.pi / 2)]),
    "SU2":  (1, [(0.0, 2 * np.pi), (0.0, np.pi), (0.0, 2 * np.pi)]),
    "BS":   (2, []),
    "PBS":  (2, []),
    "CNOT": (2, []),
    "CZ":   (2, []),
    "SWAP": (2, []),
}


def gate_to_unitary(op: GateOp, n_qubits: int) -> np.ndarray:
    """Convert GateOp to a 2^n_qubits unitary matrix."""
    n, q, p = op.name, op.qubits, op.params
    if n == "PS":
        return _embed_single(phase_shifter(p[0] if p else 0.0), q[0], n_qubits)
    if n == "HWP":
        return _embed_single(half_wave_plate(p[0] if p else np.pi / 8), q[0], n_qubits)
    if n == "QWP":
        return _embed_single(quarter_wave_plate(p[0] if p else np.pi / 8), q[0], n_qubits)
    if n == "SU2":
        return _embed_single(su2_gate(*(p + [0, 0, 0])[:3]), q[0], n_qubits)
    q1, q2 = q[0], q[1]
    if n == "BS":
        BS4 = (1 / np.sqrt(2)) * np.array(
            [[1, 0, 0, 1j], [0, 1, 1j, 0], [0, 1j, 1, 0], [1j, 0, 0, 1]],
            dtype=complex,
        )
        return _embed_two(BS4, q1, q2, n_qubits)
    if n == "PBS":
        return _embed_two(polarising_beam_splitter(), q1, q2, n_qubits)
    if n == "CNOT":
        return _embed_two(cnot_gate(), q1, q2, n_qubits)
    if n == "CZ":
        return _embed_two(cz_gate(), q1, q2, n_qubits)
    if n == "SWAP":
        return _embed_two(swap_gate(), q1, q2, n_qubits)
    raise ValueError(f"Unknown gate '{n}'")


# ---------------------------------------------------------------------------
# Circuit composition
# ---------------------------------------------------------------------------

def compute_U_total(circuit: List[GateOp], n_qubits: int) -> np.ndarray:
    """
    Compose the circuit into U_total = U_N * ... * U_2 * U_1.
    Gates are applied left-to-right (first element acts first on the state).
    """
    U = np.eye(2 ** n_qubits, dtype=complex)
    for op in circuit:
        U = gate_to_unitary(op, n_qubits) @ U
    return U


def evolve_density_matrix(
    rho_in: np.ndarray, circuit: List[GateOp], n_qubits: int
) -> np.ndarray:
    """Apply a unitary circuit: rho_out = U * rho_in * U†"""
    U = compute_U_total(circuit, n_qubits)
    return U @ rho_in @ U.conj().T


def partial_trace(rho: np.ndarray, keep: List[int], n_qubits: int) -> np.ndarray:
    """
    Partial trace: reduce to the subsystem in `keep`.
    """
    keep = sorted(keep)
    tout = [q for q in range(n_qubits) if q not in keep]
    dk, dt = 2 ** len(keep), 2 ** (n_qubits - len(keep))
    perm = keep + tout
    rho_t = rho.reshape([2] * (2 * n_qubits))
    axes = [perm[i] for i in range(n_qubits)] + [
        perm[i] + n_qubits for i in range(n_qubits)
    ]
    return np.einsum("ikjk->ij", np.transpose(rho_t, axes).reshape(dk, dt, dk, dt))


# ---------------------------------------------------------------------------
# Heralding
# ---------------------------------------------------------------------------

def herald_post_select(
    rho_full: np.ndarray,
    n_system: int,
    n_herald: int,
    herald_outcome=None,
) -> Tuple[np.ndarray, float]:
    """
    Heralded state preparation via post-selection on detector qubits.

    Returns
    -------
    rho_heralded : normalised system state conditioned on the outcome
    success_prob : probability that this outcome occurs
    """
    n_total = n_system + n_herald
    assert rho_full.shape == (2 ** n_total, 2 ** n_total), \
        f"Shape mismatch: expected ({2**n_total},{2**n_total})"
    if herald_outcome is None:
        herald_outcome = [1] * n_herald
    if len(herald_outcome) != n_herald:
        raise ValueError(
            f"herald_outcome length {len(herald_outcome)} != {n_herald}"
        )

    h_idx = int("".join(str(b) for b in herald_outcome), 2)
    h_ket = computational_basis_state(n_herald, h_idx)
    Pi_full = np.kron(
        np.eye(2 ** n_system, dtype=complex),
        np.outer(h_ket, h_ket.conj()),
    )

    rho_cond = Pi_full @ rho_full @ Pi_full.conj().T
    prob = float(np.real(np.trace(rho_cond)))

    if prob < 1e-10:
        warnings.warn(
            f"Herald outcome {herald_outcome} has probability ~0. "
            "Returning maximally-mixed state.",
            stacklevel=2,
        )
        d = 2 ** n_system
        return np.eye(d, dtype=complex) / d, 0.0

    rho_heralded = partial_trace(rho_cond / prob, list(range(n_system)), n_total)
    return rho_heralded, prob


def herald_all_outcomes(
    rho_full: np.ndarray, n_system: int, n_herald: int
) -> List[Dict]:
    """
    Enumerate all 2^n_herald herald outcomes and return the conditional
    system state and success probability for each.
    """
    results = []
    for idx in range(2 ** n_herald):
        outcome = [(idx >> (n_herald - 1 - b)) & 1 for b in range(n_herald)]
        rho_s, prob = herald_post_select(rho_full, n_system, n_herald, outcome)
        results.append(
            {
                "outcome": outcome,
                "probability": prob,
                "rho_system": rho_s,
                "purity": float(np.real(np.trace(rho_s @ rho_s))),
            }
        )
    return sorted(results, key=lambda x: x["probability"], reverse=True)


# ---------------------------------------------------------------------------
# Scaffold
# ---------------------------------------------------------------------------

def build_scaffold(n_qubits: int) -> Tuple[np.ndarray, List[str]]:
    """
    Deterministic state-preparation scaffold for 1–4 qubits.
    Returns (rho_scaffold, steps) where steps are human-readable strings.
    """
    if n_qubits not in (1, 2, 3, 4):
        raise ValueError(f"Scaffold supports 1–4 qubits, got {n_qubits}.")

    steps: List[str] = []

    # ── 1 qubit ──────────────────────────────────────────────────────────────
    if n_qubits == 1:
        bell = spdc_bell_state("phi_plus")
        rho2 = dm_from_ket(bell)
        rho1, p_her = herald_post_select(rho2, n_system=1, n_herald=1,
                                         herald_outcome=[1])
        hwp = half_wave_plate(3 * np.pi / 8)
        rho_scaffold = hwp @ rho1 @ hwp.conj().T

        steps.append(
            "Scaffold step 1: Fire one SPDC photon-pair source. "
            "The nonlinear crystal emits two polarisation-entangled photons in "
            "the Bell state Phi-plus: (horizontal-horizontal + vertical-vertical) "
            "divided by root two. The two photons travel into separate spatial modes, "
            "called signal mode and idler mode."
        )
        steps.append(
            f"Scaffold step 2: Place a single-photon threshold detector on the "
            f"idler mode and post-select on the detector clicking (outcome = 1, "
            f"meaning one photon detected). This herald event has probability "
            f"{p_her:.4f}. When the idler detector clicks, the signal photon "
            f"collapses from the entangled Bell state into a definite vertically "
            f"polarised state."
        )
        steps.append(
            "Scaffold step 3: Pass the signal photon through a half-wave plate "
            "with its fast axis at 67.5 degrees (three-eighths of pi radians) "
            "from horizontal. This rotates vertical polarisation into the diagonal "
            "superposition state: (horizontal minus vertical) divided by root two, "
            "which is the minus-eigenstate of the X Pauli operator. The system "
            "qubit is now in a clean, fully specified superposition state and is "
            "ready for the GA circuit."
        )
        return rho_scaffold, steps

    # ── 2 qubits ─────────────────────────────────────────────────────────────
    if n_qubits == 2:
        bell = spdc_bell_state("phi_plus")
        rho_scaffold = dm_from_ket(bell)

        steps.append(
            "Scaffold step 1: Fire one SPDC photon-pair source. "
            "The nonlinear crystal produces two polarisation-entangled photons "
            "in the Bell state Phi-plus: (horizontal-horizontal + vertical-vertical) "
            "divided by root two. One photon goes into qubit mode 1; the other "
            "goes into qubit mode 2. These two photons are maximally entangled: "
            "measuring one instantly determines the polarisation of the other, "
            "regardless of their separation."
        )
        steps.append(
            "Scaffold step 2: No heralding is required for two qubits. "
            "The SPDC output is itself a perfect maximally-entangled two-qubit "
            "Bell state and provides the richest possible starting resource for "
            "the optimisation: maximum entanglement, equal superposition of all "
            "same-polarisation combinations, and a density matrix with purity "
            "equal to one (a pure state)."
        )
        return rho_scaffold, steps

    # ── 3 qubits ─────────────────────────────────────────────────────────────
    if n_qubits == 3:
        bell = spdc_bell_state("phi_plus")
        psi4 = np.kron(bell, bell)
        rho4 = dm_from_ket(psi4)

        BS4 = (1 / np.sqrt(2)) * np.array(
            [[1, 0, 0, 1j], [0, 1, 1j, 0], [0, 1j, 1, 0], [1j, 0, 0, 1]],
            dtype=complex,
        )
        U4 = _embed_two(BS4, 1, 2, 4)
        rho4_after_bs = U4 @ rho4 @ U4.conj().T

        perm_idx = [0, 2, 3, 1]
        rho4_perm = _permute_qubits(rho4_after_bs, perm_idx, 4)
        rho_scaffold, p_her = herald_post_select(
            rho4_perm, n_system=3, n_herald=1, herald_outcome=[1]
        )

        steps.append(
            "Scaffold step 1: Fire two SPDC photon-pair sources simultaneously. "
            "Source 1 emits photons into modes A1 and B1 in the Bell state Phi-plus. "
            "Source 2 emits photons into modes A2 and B2 in the Bell state Phi-plus. "
            "The combined four-photon state is the tensor product of two independent "
            "Bell pairs: Phi-plus on (A1, B1) times Phi-plus on (A2, B2)."
        )
        steps.append(
            "Scaffold step 2: Route modes B1 and A2 into the two input ports of "
            "a symmetric 50:50 beam splitter. Inside the beam splitter, incoming "
            "photons undergo Hong-Ou-Mandel two-photon interference: when two "
            "indistinguishable photons enter simultaneously from opposite ports, "
            "quantum interference causes them to always exit together from the same "
            "port rather than taking separate paths. This creates entanglement "
            "between the output of the beam splitter and the unmixed modes A1 and B2."
        )
        steps.append(
            f"Scaffold step 3: Place a threshold detector on one of the beam-splitter "
            f"output ports. Post-select on that detector clicking with outcome 1 "
            f"(one photon detected). This heralding event has probability {p_her:.4f}. "
            f"The successful herald projects the remaining three modes — A1, the "
            f"other beam-splitter output, and B2 — into a three-qubit entangled "
            f"state. This is a W-class state: a superposition in which exactly one "
            f"photon is vertically polarised across the three modes, spread coherently "
            f"over all three possibilities. The three-qubit register is now ready."
        )
        return rho_scaffold, steps

    # ── 4 qubits ─────────────────────────────────────────────────────────────
    bell = spdc_bell_state("phi_plus")
    psi4 = np.kron(bell, bell)
    rho4 = dm_from_ket(psi4)

    BS4 = (1 / np.sqrt(2)) * np.array(
        [[1, 0, 0, 1j], [0, 1, 1j, 0], [0, 1j, 1, 0], [1j, 0, 0, 1]], dtype=complex
    )
    U4 = _embed_two(BS4, 1, 2, 4)
    rho_scaffold = U4 @ rho4 @ U4.conj().T

    steps.append(
        "Scaffold step 1: Fire two SPDC photon-pair sources simultaneously. "
        "Source 1 produces an entangled photon pair in modes A1 and B1 "
        "(Bell state Phi-plus: both horizontal or both vertical with equal "
        "probability). Source 2 produces a second independent entangled pair "
        "in modes A2 and B2. The initial four-photon state is the tensor "
        "product |Phi-plus> on (A1,B1) times |Phi-plus> on (A2,B2)."
    )
    steps.append(
        "Scaffold step 2: Route the two inner modes B1 and A2 into opposite "
        "input ports of a symmetric 50:50 beam splitter. The beam splitter "
        "implements the transformation: input-B1 maps to (output-C1 + i*output-C2) "
        "divided by root two, and input-A2 maps to (i*output-C1 + output-C2) "
        "divided by root two, where i is the imaginary unit representing the "
        "pi-over-two phase acquired on reflection. "
        "The four output modes are now A1, C1 (beam-splitter output 1), "
        "C2 (beam-splitter output 2), and B2."
    )
    steps.append(
        "Scaffold step 3: No heralding is required for four qubits. "
        "The beam splitter has fused the two independent Bell pairs into a "
        "four-qubit entangled state across modes (A1, C1, C2, B2) through "
        "two-photon Hong-Ou-Mandel interference at the central beam splitter. "
        "This four-qubit entangled cluster state is a substantially richer "
        "starting resource than two independent Bell pairs, because the "
        "four modes are now mutually entangled rather than pairwise entangled. "
        "The four-qubit register is ready for the GA circuit."
    )
    return rho_scaffold, steps


# ---------------------------------------------------------------------------
# Full simulation pipeline
# ---------------------------------------------------------------------------

def purity_upper_bound(rho: np.ndarray) -> float:
    """
    Maximum fidelity any pure state can achieve against rho.
    Equals the largest eigenvalue of rho.
    """
    return float(np.max(np.linalg.eigvalsh(rho)))


def simulate_experiment(
    circuit: List[GateOp],
    n_qubits: int,
    n_spdc: int = -1,
    use_vacuum: bool = False,
    n_ancilla: int = 0,
    herald_qubits: int = 0,
    herald_outcome=None,
) -> Tuple[np.ndarray, float]:
    """
    Simulate a full photonic experiment end-to-end.

    Returns
    -------
    rho_out      : output density matrix on n_qubits signal modes
    success_prob : heralding probability (1.0 if no herald qubits)
    """
    n_total = n_qubits + n_ancilla + herald_qubits

    if use_vacuum or n_spdc == 0:
        psi0 = computational_basis_state(n_total, 0)
        rho_init = dm_from_ket(psi0)
    else:
        rho_scaffold, _ = build_scaffold(n_qubits)
        extra = n_ancilla + herald_qubits
        if extra > 0:
            vac = dm_from_ket(computational_basis_state(extra, 0))
            rho_init = np.kron(rho_scaffold, vac)
        else:
            rho_init = rho_scaffold

    rho_ev = evolve_density_matrix(rho_init, circuit, n_total)

    if n_ancilla > 0:
        rho_ev = partial_trace(
            rho_ev, list(range(n_qubits + herald_qubits)), n_total
        )

    if herald_qubits > 0:
        return herald_post_select(rho_ev, n_qubits, herald_qubits, herald_outcome)
    return rho_ev, 1.0


def quantum_fidelity(rho_target: np.ndarray, rho_out: np.ndarray) -> float:
    """
    Uhlmann-Jozsa fidelity:
        F(rho, sigma) = [Tr sqrt(sqrt(rho)*sigma*sqrt(rho))]^2
    """
    try:
        tr = np.real(np.trace(rho_out))
        if tr < 1e-12:
            return 0.0
        rho_out = (rho_out / tr + rho_out.conj().T / tr) / 2
        eigs_t, vecs_t = np.linalg.eigh(rho_target)
        eigs_t = np.clip(np.real(eigs_t), 0.0, None)
        srt = (vecs_t * np.sqrt(eigs_t)) @ vecs_t.conj().T
        eigs_M = np.clip(
            np.real(np.linalg.eigvalsh(srt @ rho_out @ srt)), 0.0, None
        )
        return float(np.clip(float(np.sum(np.sqrt(eigs_M))) ** 2, 0.0, 1.0))
    except Exception:
        return 0.0

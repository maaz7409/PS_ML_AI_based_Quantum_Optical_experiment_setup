from __future__ import annotations

import random
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from library import (
    GateOp,
    GATE_REGISTRY,
    auto_n_spdc,
    build_scaffold,
    purity_upper_bound,
    quantum_fidelity,
    simulate_experiment,
)


# ---------------------------------------------------------------------------
# Gate / circuit sampling
# ---------------------------------------------------------------------------

SQ_GATES = ["PS", "HWP", "QWP", "SU2"]
TQ_GATES = ["BS", "PBS", "CNOT", "CZ", "SWAP"]


def _random_gate(n_qubits: int, rng: random.Random) -> GateOp:
    """Sample one gate uniformly from the full component library."""
    name = rng.choice(SQ_GATES + (TQ_GATES if n_qubits >= 2 else []))
    arity, prs = GATE_REGISTRY[name]
    qs = rng.sample(range(n_qubits), 2) if arity == 2 else [rng.randint(0, n_qubits - 1)]
    return GateOp(name, qs, [np.random.uniform(lo, hi) for lo, hi in prs])


def _random_circuit(nq: int, L: int, rng: random.Random) -> List[GateOp]:
    return [_random_gate(nq, rng) for _ in range(L)]


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """One candidate circuit in the GA population."""
    circuit: List[GateOp]
    fitness: float = -1.0

    def copy(self):
        return Individual([g.copy() for g in self.circuit], self.fitness)


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _crossover(
    p1: Individual, p2: Individual, rate: float, rng: random.Random
) -> Tuple[Individual, Individual]:
    """Uniform crossover: each gate slot independently exchanged with prob 0.5."""
    if rng.random() > rate:
        return p1.copy(), p2.copy()
    L = min(len(p1.circuit), len(p2.circuit))
    if L < 2:
        return p1.copy(), p2.copy()
    c1, c2 = [], []
    for i in range(L):
        if rng.random() < 0.5:
            c1.append(p1.circuit[i].copy())
            c2.append(p2.circuit[i].copy())
        else:
            c1.append(p2.circuit[i].copy())
            c2.append(p1.circuit[i].copy())
    if len(p1.circuit) > L:
        c1 += [g.copy() for g in p1.circuit[L:]]
    if len(p2.circuit) > L:
        c2 += [g.copy() for g in p2.circuit[L:]]
    return Individual(c1), Individual(c2)


def _mutate(
    ind: Individual, nq: int, mr: float, ps: float, mlen: int, rng: random.Random
) -> Individual:
    """Five stochastic mutation operators per call."""
    ind = ind.copy()
    circ = ind.circuit
    for i in range(len(circ)):
        if rng.random() < mr * 0.4:
            circ[i] = _random_gate(nq, rng)
        elif rng.random() < mr * 0.6:
            for k, (lo, hi) in enumerate(GATE_REGISTRY[circ[i].name][1]):
                v = circ[i].params[k] + np.random.normal(0, ps)
                circ[i].params[k] = lo + (v - lo) % (hi - lo)
    if rng.random() < mr * 0.5 and len(circ) < mlen:
        circ.insert(rng.randint(0, len(circ)), _random_gate(nq, rng))
    if rng.random() < mr * 0.4 and len(circ) > 1:
        circ.pop(rng.randint(0, len(circ) - 1))
    if rng.random() < mr * 0.3:
        idx = rng.randint(0, len(circ) - 1)
        g = circ[idx]
        g.qubits = (
            rng.sample(range(nq), 2)
            if GATE_REGISTRY[g.name][0] == 2 and nq >= 2
            else [rng.randint(0, nq - 1)]
        )
    ind.circuit = circ
    return ind


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def _eval(ind, rho_t, nq, ns, na, hq=0, ho=None):
    """Evaluate fidelity of one circuit against the target."""
    try:
        rho_out, _ = simulate_experiment(
            ind.circuit, nq, ns, n_ancilla=na,
            herald_qubits=hq, herald_outcome=ho,
        )
        return quantum_fidelity(rho_t, rho_out)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Local parameter optimisation
# ---------------------------------------------------------------------------

def _golden_section_search(
    f_scalar, lo: float, hi: float, tol: float = 1e-4, max_iter: int = 60
) -> float:
    """
    Golden-section search for the maximum of a scalar function on [lo, hi].
    """
    phi_g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - phi_g * (b - a)
    d = a + phi_g * (b - a)
    fc, fd = f_scalar(c), f_scalar(d)

    for _ in range(max_iter):
        if (b - a) < tol:
            break
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + phi_g * (b - a)
            fd = f_scalar(d)
        else:
            b, d, fd = d, c, fc
            c = b - phi_g * (b - a)
            fc = f_scalar(c)

    return (a + b) / 2


def _local_param_opt(
    ind, rho_t, nq, ns, na, hq=0, ho=None,
    n_restarts: int = 3, max_iters: int = 40,
    lr: float = 0.15, use_gss: bool = True,
):
    """Coordinate-ascent optimisation of all continuous gate parameters."""
    best = ind.copy()
    best_f = _eval(best, rho_t, nq, ns, na, hq, ho)

    for _ in range(n_restarts):
        current = best.copy()
        for _ in range(max_iters):
            improved = False
            for i, gate in enumerate(current.circuit):
                for k, (lo, hi) in enumerate(GATE_REGISTRY[gate.name][1]):
                    span = hi - lo
                    if use_gss:
                        def _f(x, _i=i, _k=k):
                            current.circuit[_i].params[_k] = x
                            return _eval(current, rho_t, nq, ns, na, hq, ho)
                        x_opt = _golden_section_search(_f, lo, hi)
                        current.circuit[i].params[k] = x_opt
                        f_new = _eval(current, rho_t, nq, ns, na, hq, ho)
                    else:
                        orig = gate.params[k]
                        f_new = best_f
                        for delta in (lr, -lr, lr * 0.3, -lr * 0.3):
                            gate.params[k] = lo + (orig + delta - lo) % span
                            f = _eval(current, rho_t, nq, ns, na, hq, ho)
                            if f > f_new:
                                f_new = f
                                x_opt = gate.params[k]
                        gate.params[k] = (
                            x_opt if f_new > best_f else best.circuit[i].params[k]
                        )

                    if f_new > best_f:
                        best_f = f_new
                        best = current.copy()
                        improved = True

            if not improved:
                break

    best.fitness = best_f
    return best


# ---------------------------------------------------------------------------
# Gate narration helper
# ---------------------------------------------------------------------------

def _narrate_gate(step: int, gate, n_system: int, n_ancilla: int) -> str:
    """
    Produce a fully hardcoded, human-readable English sentence for a single
    gate operation.
    """
    _role: Dict[int, str] = {}

    if n_system == 1:
        _role[0] = "the first system qubit"
        if n_ancilla >= 1: _role[1] = "the first ancilla qubit"
        if n_ancilla >= 2: _role[2] = "the second ancilla qubit"
        if n_ancilla >= 3: _role[3] = "the third ancilla qubit"
        if n_ancilla >= 4: _role[4] = "the fourth ancilla qubit"
    elif n_system == 2:
        _role[0] = "the first system qubit"
        _role[1] = "the second system qubit"
        if n_ancilla >= 1: _role[2] = "the first ancilla qubit"
        if n_ancilla >= 2: _role[3] = "the second ancilla qubit"
        if n_ancilla >= 3: _role[4] = "the third ancilla qubit"
        if n_ancilla >= 4: _role[5] = "the fourth ancilla qubit"
    elif n_system == 3:
        _role[0] = "the first system qubit"
        _role[1] = "the second system qubit"
        _role[2] = "the third system qubit"
        if n_ancilla >= 1: _role[3] = "the first ancilla qubit"
        if n_ancilla >= 2: _role[4] = "the second ancilla qubit"
        if n_ancilla >= 3: _role[5] = "the third ancilla qubit"
    elif n_system >= 4:
        _role[0] = "the first system qubit"
        _role[1] = "the second system qubit"
        _role[2] = "the third system qubit"
        _role[3] = "the fourth system qubit"
        if n_ancilla >= 1: _role[4] = "the first ancilla qubit"
        if n_ancilla >= 2: _role[5] = "the second ancilla qubit"
        if n_ancilla >= 3: _role[6] = "the third ancilla qubit"
        if n_ancilla >= 4: _role[7] = "the fourth ancilla qubit"

    def _role_name(idx: int) -> str:
        return _role.get(idx, f"qubit number {idx + 1}")

    def _phi_words(phi: float) -> str:
        deg = np.degrees(phi)
        pi_frac = phi / np.pi
        nice = {
            0.00: "zero radians (zero degrees)",
            0.25: "one quarter pi radians (45.00 degrees)",
            0.50: "one half pi radians (90.00 degrees)",
            0.75: "three quarters pi radians (135.00 degrees)",
            1.00: "pi radians (180.00 degrees)",
            1.25: "five quarters pi radians (225.00 degrees)",
            1.50: "three halves pi radians (270.00 degrees)",
            1.75: "seven quarters pi radians (315.00 degrees)",
            2.00: "two pi radians (360.00 degrees)",
        }
        for frac, label in nice.items():
            if abs(pi_frac - frac) < 0.005:
                return label
        return (
            f"{phi:.6f} radians, which is {deg:.3f} degrees "
            f"or equivalently {pi_frac:.4f} times pi"
        )

    def _theta_words(theta: float) -> str:
        deg = np.degrees(theta)
        pi_frac = theta / np.pi
        nice = {
            0.00: "zero radians (zero degrees, fast axis horizontal)",
            0.25: "one quarter pi radians (45.00 degrees from horizontal)",
            0.50: "one half pi radians (90.00 degrees, fast axis vertical)",
        }
        for frac, label in nice.items():
            if abs(pi_frac - frac) < 0.005:
                return label
        return (
            f"{theta:.6f} radians, which is {deg:.3f} degrees from horizontal "
            f"(equivalently {pi_frac:.4f} times pi)"
        )

    def _euler_words(angle: float, axis_name: str) -> str:
        deg = np.degrees(angle)
        pi_frac = angle / np.pi
        nice = {
            0.00: "zero radians (no rotation)",
            0.50: "one half pi radians (90.00 degrees)",
            1.00: "pi radians (180.00 degrees, a complete half-turn)",
            1.50: "three halves pi radians (270.00 degrees)",
            2.00: "two pi radians (360.00 degrees, a full turn back to start)",
        }
        for frac, label in nice.items():
            if abs(pi_frac - frac) < 0.005:
                return f"angle {axis_name} equal to {label}"
        return (
            f"angle {axis_name} equal to {angle:.6f} radians "
            f"({deg:.3f} degrees, or {pi_frac:.4f} times pi)"
        )

    g = gate.name
    q = gate.qubits
    p = gate.params
    s = step

    # ── PS ────────────────────────────────────────────────────────────────────
    if g == "PS":
        phi = p[0]
        phi_text = _phi_words(phi)
        q0 = q[0]
        n0 = _role_name(q0)
        print(f"Step {s}: Phase Shifter applied to {n0}.")
        print(f"  The phase angle phi is set to {phi_text}.")
        print(f"  This component leaves the horizontal polarisation (the zero-state) completely unchanged.")
        print(f"  The vertical polarisation (the one-state) acquires an additional phase advance of {phi_text}.")
        print(f"  Physically this is a birefringent crystal or electro-optic modulator.")
        return f"Step {s}: Phase Shifter on {n0} — phi = {phi_text}."

    # ── HWP ───────────────────────────────────────────────────────────────────
    if g == "HWP":
        theta = p[0]
        deg = float(np.degrees(theta))
        theta_text = _theta_words(theta)
        rotation = 2 * deg
        n0 = _role_name(q[0])
        print(f"Step {s}: Half-Wave Plate on {n0}.")
        print(f"  The fast axis is physically oriented at {deg:.3f} degrees from horizontal ({theta_text}).")
        print(f"  Introduces pi radians of retardation; net polarisation rotation is {rotation:.3f} degrees.")
        if abs(deg - 45.0) < 0.5:
            print(f"  At 45 degrees this plate acts as a bit-flip: swaps horizontal and vertical polarisation.")
        return f"Step {s}: Half-Wave Plate on {n0} — fast axis at {deg:.3f} degrees ({theta_text})."

    # ── QWP ───────────────────────────────────────────────────────────────────
    if g == "QWP":
        theta = p[0]
        deg = float(np.degrees(theta))
        theta_text = _theta_words(theta)
        n0 = _role_name(q[0])
        print(f"Step {s}: Quarter-Wave Plate on {n0}.")
        print(f"  The fast axis is at {deg:.3f} degrees from horizontal ({theta_text}).")
        print(f"  Introduces pi/2 retardation; converts linear to elliptical polarisation (or vice versa).")
        if abs(deg - 45.0) < 0.5:
            print(f"  At 45 degrees: converts horizontal linear to right-hand circular polarisation.")
        return f"Step {s}: Quarter-Wave Plate on {n0} — fast axis at {deg:.3f} degrees ({theta_text})."

    # ── SU2 ───────────────────────────────────────────────────────────────────
    if g == "SU2":
        alpha, beta, gamma = p[0], p[1], p[2]
        n0 = _role_name(q[0])
        print(f"Step {s}: Universal Single-Qubit Rotation (SU2) on {n0}.")
        print(f"  ZYZ Euler decomposition: three rotations in sequence.")
        print(f"  First rotation  — Z axis: {_euler_words(alpha, 'alpha')}.")
        print(f"  Second rotation — Y axis: {_euler_words(beta, 'beta')}.")
        print(f"  Third rotation  — Z axis: {_euler_words(gamma, 'gamma')}.")
        print(f"  Physical implementation: QWP → HWP → QWP sequence on the beam path.")
        return (
            f"Step {s}: SU2 rotation on {n0} — "
            f"alpha = {alpha:.4f} rad, beta = {beta:.4f} rad, gamma = {gamma:.4f} rad."
        )

    # ── BS ────────────────────────────────────────────────────────────────────
    if g == "BS":
        q1, q2 = q[0], q[1]
        n1, n2 = _role_name(q1), _role_name(q2)
        print(f"Step {s}: Symmetric 50:50 Beam Splitter between {n1} and {n2}.")
        print(
            f"  Each incoming photon has equal probability of transmission (amplitude 1/√2) "
            f"or reflection (amplitude i/√2)."
        )
        print(
            f"  Hong-Ou-Mandel two-photon interference occurs when both ports are "
            f"simultaneously occupied by indistinguishable photons."
        )
        return f"Step {s}: Beam Splitter between {n1} and {n2}."

    # ── PBS ───────────────────────────────────────────────────────────────────
    if g == "PBS":
        q1, q2 = q[0], q[1]
        n1, n2 = _role_name(q1), _role_name(q2)
        print(f"Step {s}: Polarising Beam Splitter between {n1} and {n2}.")
        print(
            f"  Horizontal photons transmit; vertical photons reflect. "
            f"Equivalent to a partial SWAP on the polarisation degree of freedom."
        )
        print(f"  Key element in linear-optic Bell-state measurement and entanglement-swapping circuits.")
        return f"Step {s}: Polarising Beam Splitter between {n1} and {n2}."

    # ── CNOT ──────────────────────────────────────────────────────────────────
    if g == "CNOT":
        q1, q2 = q[0], q[1]
        n1, n2 = _role_name(q1), _role_name(q2)
        print(f"Step {s}: Controlled-NOT Gate. Control: {n1}. Target: {n2}.")
        print(
            f"  If the control photon is vertically polarised (|1>), the target "
            f"photon's polarisation is flipped; otherwise the target is unchanged."
        )
        print(
            f"  Physical note: requires ancilla photons and KLM post-selection "
            f"in linear optics. Treated here as an ideal unitary."
        )
        return f"Step {s}: CNOT gate — control = {n1}, target = {n2}."

    # ── CZ ────────────────────────────────────────────────────────────────────
    if g == "CZ":
        q1, q2 = q[0], q[1]
        n1, n2 = _role_name(q1), _role_name(q2)
        print(f"Step {s}: Controlled-Z Gate acting on {n1} and {n2}.")
        print(
            f"  Applies a −1 phase factor only when both qubits are in the |1> state "
            f"(both photons vertically polarised). Symmetric: either qubit may be the control."
        )
        return f"Step {s}: CZ gate between {n1} and {n2}."

    # ── SWAP ──────────────────────────────────────────────────────────────────
    if g == "SWAP":
        q1, q2 = q[0], q[1]
        n1, n2 = _role_name(q1), _role_name(q2)
        print(f"Step {s}: SWAP Gate between {n1} and {n2}.")
        print(
            f"  Exchanges the complete quantum states of the two qubits. "
            f"Perfectly reversible — no measurement or information loss."
        )
        print(
            f"  Physical options: crossed optical path, or three CNOT gates "
            f"(CNOT_12, CNOT_21, CNOT_12)."
        )
        return f"Step {s}: SWAP gate between {n1} and {n2}."

    # ── fallback ──────────────────────────────────────────────────────────────
    print(f"Step {s}: Gate '{g}' on qubits {q} — no hardcoded description available.")
    return f"Step {s}: {g} on {[_role_name(i) for i in q]}."


# ---------------------------------------------------------------------------
# QuantumOpticsGA
# ---------------------------------------------------------------------------

class QuantumOpticsGA:
    """
    Genetic Algorithm for autonomous photonic circuit design.
    """

    def __init__(
        self,
        rho_target,
        n_qubits,
        n_spdc=-1,
        n_ancilla=-1,
        herald_qubits=0,
        herald_outcome=None,
        population_size=80,
        init_circuit_length=8,
        max_circuit_length=20,
        max_generations=300,
        crossover_rate=0.75,
        base_mutation_rate=0.25,
        param_sigma=0.25,
        elitism_k=6,
        tournament_size=5,
        fidelity_target=0.995,
        local_opt=True,
        local_opt_freq=15,
        seed=0,
        verbose=True,
    ):
        self.rho_target = rho_target
        self.n_qubits = n_qubits
        self.herald_qubits = herald_qubits
        self.herald_outcome = herald_outcome
        self.population_size = population_size
        self.init_circuit_length = init_circuit_length
        self.max_circuit_length = max_circuit_length
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = base_mutation_rate
        self.base_mutation_rate = base_mutation_rate
        self.param_sigma = param_sigma
        self.elitism_k = elitism_k
        self.tournament_size = tournament_size
        self.fidelity_target = fidelity_target
        self.local_opt = local_opt
        self.local_opt_freq = local_opt_freq
        self.verbose = verbose

        tp = float(np.real(np.trace(rho_target @ rho_target)))
        self.target_purity = tp
        if n_ancilla == -1:
            if tp > 1.0 - 1e-4:
                self.n_ancilla = 0
            else:
                self.n_ancilla = n_qubits
                if verbose:
                    ub = purity_upper_bound(rho_target)
                    print(
                        f"  [mixed target]  purity={tp:.4f}  "
                        f"ancilla={n_qubits}  ceiling={ub:.4f}  "
                        f"fidelity_target -> {min(fidelity_target, ub - 0.01):.4f}"
                    )
                    self.fidelity_target = min(fidelity_target, ub - 0.01)
        else:
            self.n_ancilla = n_ancilla

        self.n_total = n_qubits + self.n_ancilla + herald_qubits
        self.n_spdc = auto_n_spdc(self.n_total) if n_spdc == -1 else n_spdc

        random.seed(seed)
        np.random.seed(seed)
        self._rng = random.Random(seed)
        self.history = {
            "generation": [], "best_fidelity": [], "avg_fidelity": [],
            "circuit_length": [], "mutation_rate": [], "best_individual": [],
            "raw_gen0_best_fidelity": None, "raw_gen0_avg_fidelity": None,
            "param_trajectory": [],
        }
        self.best = None

    # ── Internal helpers ────────────────────────────────────────────────────

    def _ei(self, ind):
        return _eval(
            ind, self.rho_target, self.n_qubits,
            self.n_spdc, self.n_ancilla,
            self.herald_qubits, self.herald_outcome,
        )

    def _eval_pop(self, pop):
        for ind in pop:
            if ind.fitness < 0:
                ind.fitness = self._ei(ind)

    def _tournament(self, pop):
        return max(
            self._rng.sample(pop, min(self.tournament_size, len(pop))),
            key=lambda x: x.fitness,
        ).copy()

    def _stagnating(self, w=25):
        bf = self.history["best_fidelity"]
        return len(bf) >= w and max(bf[-w:]) - min(bf[-w:]) < 3e-4

    # ── Main evolution loop ─────────────────────────────────────────────────

    def evolve(self) -> Individual:
        """Run the genetic algorithm. Returns the best Individual found."""
        t0 = time.time()
        pop = [
            Individual(_random_circuit(self.n_total, self.init_circuit_length, self._rng))
            for _ in range(self.population_size)
        ]
        self._eval_pop(pop)

        pop.sort(key=lambda x: x.fitness, reverse=True)
        self.history["raw_gen0_best_fidelity"] = pop[0].fitness
        self.history["raw_gen0_avg_fidelity"] = float(np.mean([i.fitness for i in pop]))

        if self.verbose:
            print(f"\n{'='*60}")
            print(
                f"  {self.n_qubits}q | spdc={self.n_spdc} ancilla={self.n_ancilla} "
                f"herald={self.herald_qubits} | pop={self.population_size} "
                f"gen={self.max_generations} | baseline={pop[0].fitness:.4f}"
            )
            print(f"{'='*60}")
            print(f"  {'gen':>4}  {'best_F':>9}  {'avg_F':>9}  {'depth':>6}  {'mut':>7}  {'time':>7}")
            print(f"  {'-'*56}")

        for gen in range(self.max_generations):
            pop.sort(key=lambda x: x.fitness, reverse=True)
            best_ind = pop[0]

            if self.local_opt and gen > 0 and gen % self.local_opt_freq == 0:
                for idx in range(min(self.elitism_k, len(pop))):
                    pop[idx] = _local_param_opt(
                        pop[idx], self.rho_target, self.n_qubits, self.n_spdc,
                        self.n_ancilla, self.herald_qubits, self.herald_outcome,
                    )
                pop.sort(key=lambda x: x.fitness, reverse=True)
                best_ind = pop[0]

            avg_f = float(np.mean([i.fitness for i in pop]))
            self.history["generation"].append(gen)
            self.history["best_fidelity"].append(best_ind.fitness)
            self.history["avg_fidelity"].append(avg_f)
            self.history["circuit_length"].append(len(best_ind.circuit))
            self.history["mutation_rate"].append(self.mutation_rate)
            self.history["best_individual"].append(best_ind.copy())

            snapshot = {}
            for step, gate in enumerate(best_ind.circuit, 1):
                if gate.params:
                    snapshot[step] = {
                        "gate": gate.name,
                        "qubits": gate.qubits.copy(),
                        "params": gate.params.copy(),
                    }
            self.history["param_trajectory"].append(snapshot)

            if self.best is None or best_ind.fitness > self.best.fitness:
                self.best = best_ind.copy()

            if self.verbose and gen % 5 == 0:
                print(
                    f"  {gen:>4}  {best_ind.fitness:>9.6f}  {avg_f:>9.6f}  "
                    f"{len(best_ind.circuit):>6}  {self.mutation_rate:>7.4f}  "
                    f"{time.time()-t0:>6.1f}s"
                )

            if best_ind.fitness >= self.fidelity_target:
                if self.verbose:
                    print(f"  [converged at gen {gen}  F={best_ind.fitness:.6f}]")
                break

            self.mutation_rate = (
                min(0.75, self.mutation_rate * 1.15)
                if self._stagnating()
                else max(self.base_mutation_rate, self.mutation_rate * 0.97)
            )

            next_pop = [ind.copy() for ind in pop[: self.elitism_k]]
            while len(next_pop) < self.population_size:
                c1, c2 = _crossover(
                    self._tournament(pop), self._tournament(pop),
                    self.crossover_rate, self._rng,
                )
                for c in (c1, c2):
                    c = _mutate(
                        c, self.n_total, self.mutation_rate,
                        self.param_sigma, self.max_circuit_length, self._rng,
                    )
                    if len(next_pop) < self.population_size:
                        next_pop.append(c)
            self._eval_pop(next_pop[self.elitism_k:])
            pop = next_pop

        if self.local_opt and self.best is not None:
            self.best = _local_param_opt(
                self.best, self.rho_target, self.n_qubits, self.n_spdc,
                self.n_ancilla, self.herald_qubits, self.herald_outcome,
                n_restarts=5, max_iters=80, lr=0.08,
            )
        if self.verbose:
            print(f"  {'-'*56}")
            print(f"  done  {time.time()-t0:.1f}s  |  F={self.best.fitness:.8f}")
        return self.best

    # ── Output helpers ──────────────────────────────────────────────────────

    def get_output_state(self, individual=None) -> Tuple[np.ndarray, float]:
        """Return (rho_out, success_prob) for the given individual."""
        ind = individual or self.best
        return simulate_experiment(
            ind.circuit, self.n_qubits, self.n_spdc,
            n_ancilla=self.n_ancilla,
            herald_qubits=self.herald_qubits,
            herald_outcome=self.herald_outcome,
        )

    def print_report(self):
        """Print a concise summary of the best circuit found."""
        ind = self.best
        if ind is None:
            return
        rho_out, prob = self.get_output_state(ind)
        fid = quantum_fidelity(self.rho_target, rho_out)
        raw0 = self.history.get("raw_gen0_best_fidelity")
        td = 0.5 * float(
            np.sum(np.abs(np.linalg.eigvalsh(self.rho_target - rho_out)))
        )

        print(f"\n{'='*60}")
        print(
            f"  fidelity     : {fid:.8f}"
            + (f"  (+{fid-raw0:.6f} vs baseline)" if raw0 else "")
        )
        print(
            f"  trace dist   : {td:.6f}  |  "
            f"max |Δρ| : {np.abs(self.rho_target-rho_out).max():.6f}"
        )
        if self.herald_qubits > 0:
            print(
                f"  herald prob  : {prob:.6f}  "
                f"outcome={self.herald_outcome or [1]*self.herald_qubits}"
            )
        if self.n_ancilla > 0:
            print(
                f"  ceiling      : {purity_upper_bound(self.rho_target):.6f}  "
                f"(target purity={self.target_purity:.4f})"
            )
        print(
            f"  circuit      : {len(ind.circuit)} gates  |  "
            f"{self.n_qubits}q sys  {self.n_ancilla}q anc  "
            f"{self.herald_qubits}q herald  {self.n_spdc} SPDC"
        )
        print(f"\n  {'step':<6}  gate")
        print(f"  {'-'*40}")
        for i, g in enumerate(ind.circuit, 1):
            print(f"  {i:<6}  {g}")
        print(f"{'='*60}")

    def evaluate_on_test_set(self, test_states, individual=None, verbose=True):
        """Evaluate a fixed circuit on held-out states. Returns fidelity statistics."""
        ind = individual or self.best
        if ind is None:
            raise RuntimeError("Call evolve() first.")
        fids = []
        for rho_t in test_states:
            ro, _ = simulate_experiment(
                ind.circuit, self.n_qubits, self.n_spdc,
                n_ancilla=self.n_ancilla,
                herald_qubits=self.herald_qubits,
                herald_outcome=self.herald_outcome,
            )
            fids.append(quantum_fidelity(rho_t, ro))
        res = {
            "fidelities": fids,
            "mean": float(np.mean(fids)),
            "std": float(np.std(fids)),
            "min": float(np.min(fids)),
            "max": float(np.max(fids)),
        }
        if verbose:
            gap = self.best.fitness - res["mean"]
            print(
                f"\n  test ({len(test_states)} states)  "
                f"mean={res['mean']:.6f}  std={res['std']:.4f}  "
                f"min={res['min']:.4f}  max={res['max']:.4f}  gap={gap:+.4f}"
            )
        return res

    # ── Parameter inspection ────────────────────────────────────────────────

    def param_table(self) -> List[Dict]:
        """Return parameters of every gate in the best circuit as plain dicts."""
        _labels = {
            "PS":  ["phi"],
            "HWP": ["theta"],
            "QWP": ["theta"],
            "SU2": ["alpha", "beta", "gamma"],
        }
        rows = []
        if self.best is None:
            return rows
        for step, gate in enumerate(self.best.circuit, 1):
            if not gate.params:
                continue
            rows.append({
                "step":   step,
                "gate":   gate.name,
                "qubits": gate.qubits.copy(),
                "params": gate.params.copy(),
                "labels": _labels.get(gate.name, [f"p{i}" for i in range(len(gate.params))]),
            })
        return rows

    def print_params(self):
        """Print a compact table of all continuous gate parameters in the best circuit."""
        rows = self.param_table()
        if not rows:
            print("  No parameterised gates in best circuit.")
            return
        print(f"\n  {'step':<5}  {'gate':<5}  {'qubits':<10}  parameters")
        print(f"  {'-'*55}")
        for r in rows:
            vals = "  ".join(
                f"{v:7.4f} ({v/np.pi:+.4f}π)  [{r['labels'][i]}]"
                for i, v in enumerate(r["params"])
            )
            print(f"  {r['step']:<5}  {r['gate']:<5}  {str(r['qubits']):<10}  {vals}")

    def describe_experiment(self, save_path: Optional[str] = None) -> str:
        """Produce a fully human-readable description of the best circuit found."""
        if self.best is None:
            return "No circuit has been found yet. Run evolve() first."

        ind = self.best
        circuit = ind.circuit
        rho_out, prob = self.get_output_state(ind)
        fidelity = quantum_fidelity(self.rho_target, rho_out)
        raw0 = self.history.get("raw_gen0_best_fidelity") or 0.0
        trace_dist = 0.5 * float(
            np.sum(np.abs(np.linalg.eigvalsh(self.rho_target - rho_out)))
        )
        purity_out = float(np.real(np.trace(rho_out @ rho_out)))

        def _ordinal(n: int) -> str:
            suffixes = {1: "st", 2: "nd", 3: "rd"}
            return f"{n}{suffixes.get(n if n < 20 else n % 10, 'th')}"

        def _qubit_name(idx: int) -> str:
            if idx < self.n_qubits:
                return f"system qubit {_ordinal(idx + 1)}"
            anc_idx = idx - self.n_qubits
            if anc_idx < self.n_ancilla:
                return f"ancilla qubit {_ordinal(anc_idx + 1)}"
            her_idx = anc_idx - self.n_ancilla
            return f"herald qubit {_ordinal(her_idx + 1)}"

        def _describe_gate(step_number: int, gate) -> str:
            return _narrate_gate(step_number, gate, self.n_qubits, self.n_ancilla)

        lines: List[str] = []

        lines.append("=" * 72)
        lines.append("QUANTUM OPTICAL EXPERIMENT DESCRIPTION")
        lines.append("Generated automatically from the best circuit found by the")
        lines.append("genetic algorithm optimisation.")
        lines.append("=" * 72)

        _, scaffold_steps = build_scaffold(self.n_qubits)
        lines.append("")
        lines.append("PART 1 — STATE PREPARATION SCAFFOLD")
        lines.append("-" * 40)
        lines.append(
            f"Before the optimised circuit runs, a fixed preparation scaffold "
            f"builds the {self.n_qubits}-qubit entangled starting register from "
            f"raw SPDC photon pairs."
        )
        lines.append("")
        for s in scaffold_steps:
            lines.append(s)
            lines.append("")

        lines.append("")
        lines.append("PART 2 — SYSTEM OVERVIEW")
        lines.append("-" * 40)
        lines.append(
            f"The experiment operates on {self.n_qubits} system "
            f"{'qubit' if self.n_qubits == 1 else 'qubits'}. "
            f"Each qubit is encoded in the polarisation of a single photon."
        )
        spdc_desc = {
            1: "one SPDC photon-pair source",
            2: "two SPDC photon-pair sources",
            3: "three SPDC photon-pair sources",
        }
        lines.append(
            f"The photon source consists of "
            f"{spdc_desc.get(self.n_spdc, f'{self.n_spdc} SPDC sources')}. "
            f"Each SPDC source produces a Bell state Phi-plus."
        )
        if self.n_ancilla > 0:
            lines.append(
                f"The circuit also uses {self.n_ancilla} ancilla "
                f"{'qubit' if self.n_ancilla == 1 else 'qubits'} "
                f"that are entangled with the system and then traced out to produce "
                f"a mixed output state."
            )
        if self.herald_qubits > 0:
            outcome_str = ", ".join(
                ("one (detector clicks)" if b == 1 else "zero (detector dark)")
                for b in (self.herald_outcome or [1] * self.herald_qubits)
            )
            lines.append(
                f"The experiment uses {self.herald_qubits} herald detector(s) "
                f"post-selecting on [{outcome_str}]. Success probability: {prob:.4f}."
            )
        lines.append(f"Total qubits in simulation: {self.n_total}.")

        lines.append("")
        lines.append("PART 3 — TARGET QUANTUM STATE")
        lines.append("-" * 40)
        purity_target = self.target_purity
        lines.append(
            f"The goal is to prepare a specific quantum state on "
            f"{self.n_qubits} {'qubit' if self.n_qubits == 1 else 'qubits'}."
        )
        if abs(purity_target - 1.0) < 1e-4:
            lines.append("The target state is a pure state (purity = 1.0).")
        else:
            lines.append(f"The target state is mixed with purity {purity_target:.6f}.")
            ceiling = purity_upper_bound(self.rho_target)
            lines.append(
                f"Maximum achievable fidelity (largest eigenvalue of target): {ceiling:.6f}."
            )

        lines.append("")
        lines.append("PART 4 — PERFORMANCE OF THE FOUND CIRCUIT")
        lines.append("-" * 40)
        lines.append(f"Fidelity with target: {fidelity:.8f}.")
        lines.append(f"Trace distance from target: {trace_dist:.8f}.")
        lines.append(f"Purity of the output state: {purity_out:.6f}.")
        lines.append(
            f"GA started at baseline fidelity {raw0:.6f}. "
            f"Final fidelity {fidelity:.6f} (+{fidelity - raw0:.6f})."
        )

        lines.append("")
        lines.append("PART 5 — STEP-BY-STEP OPTIMISED CIRCUIT")
        lines.append("-" * 40)
        lines.append(
            f"The circuit consists of {len(circuit)} optical operations applied in sequence."
        )
        lines.append("")
        for step_number, gate in enumerate(circuit, 1):
            lines.append(_describe_gate(step_number, gate))
            lines.append("")

        lines.append("PART 6 — HOW TO READ THE OUTPUT STATE")
        lines.append("-" * 40)
        dim = 2 ** self.n_qubits
        basis_labels = [
            "|"
            + "".join(
                ("horizontal" if bit == "0" else "vertical")
                + (" " if idx < self.n_qubits - 1 else "")
                for idx, bit in enumerate(bin(i)[2:].zfill(self.n_qubits))
            )
            + ">"
            for i in range(dim)
        ]
        lines.append(
            f"The output state is a {dim}×{dim} density matrix. "
            f"Basis states: " + ", ".join(basis_labels) + ". "
            f"Diagonal entries = measurement probabilities (sum to 1). "
            f"Off-diagonal entries = quantum coherences."
        )
        lines.append("")
        lines.append("=" * 72)

        full_text = "\n".join(lines)
        print(full_text)
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(full_text)
        return full_text

    # ── Plotting ────────────────────────────────────────────────────────────

    def plot_param_trajectories(
        self,
        steps: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ):
        """Plot how each gate's parameters evolved over generations."""
        traj = self.history["param_trajectory"]
        gens = self.history["generation"]
        if not traj:
            print("  No trajectory data — call evolve() first.")
            return

        all_steps = sorted({step for snap in traj for step in snap})
        if steps is not None:
            all_steps = [s for s in all_steps if s in steps]
        if not all_steps:
            print("  No parameterised steps found for the requested indices.")
            return

        n_panels = len(all_steps)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)
        fig.suptitle(
            f"Parameter trajectories  |  {self.n_qubits}-qubit circuit  |  "
            f"F(final)={self.best.fitness:.4f}",
            fontsize=11, fontweight="bold",
        )

        _param_labels = {
            "PS":  ["phi"],
            "HWP": ["theta"],
            "QWP": ["theta"],
            "SU2": ["alpha", "beta", "gamma"],
        }
        _colors = ["royalblue", "tomato", "seagreen"]

        for ax, step in zip(axes[0], all_steps):
            segments: Dict[str, Tuple[List[int], List[List[float]]]] = {}
            for g_idx, snap in enumerate(traj):
                if step not in snap:
                    continue
                entry = snap[step]
                gate_name = entry["gate"]
                if gate_name not in segments:
                    segments[gate_name] = ([], [])
                segments[gate_name][0].append(gens[g_idx])
                segments[gate_name][1].append(entry["params"])

            for gate_name, (gen_list, param_list) in segments.items():
                labels = _param_labels.get(
                    gate_name, [f"p{i}" for i in range(len(param_list[0]))]
                )
                params_T = list(zip(*param_list))
                for p_idx, (param_vals, label) in enumerate(zip(params_T, labels)):
                    ax.plot(
                        gen_list, param_vals,
                        color=_colors[p_idx % len(_colors)],
                        lw=1.8, label=f"{gate_name}  {label}",
                    )

            ax.set_title(f"step {step}", fontsize=10)
            ax.set_xlabel("generation")
            ax.set_ylabel("angle (rad)")
            ax.set_ylim([-0.15, 2 * np.pi + 0.15])
            ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            ax.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"], fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.3)
            if len(gens) > 1:
                ax.set_xlim([gens[0] - 0.5, gens[-1] + 0.5])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot(self, save_path=None):
        """Three-panel evolution plot: fidelity history, circuit depth, mutation rate."""
        gen, bf, af, cl, mr = (
            self.history[k]
            for k in ["generation", "best_fidelity", "avg_fidelity",
                      "circuit_length", "mutation_rate"]
        )
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(
            f"GA Evolution  |  {self.n_qubits}-qubit target  |  "
            f"purity={self.target_purity:.3f}  |  n_spdc={self.n_spdc}  |  "
            f"n_ancilla={self.n_ancilla}",
            fontsize=11, fontweight="bold",
        )

        def _p(ax, x, y, **kw):
            (
                ax.scatter(x, y, s=50, zorder=5, color=kw.get("color", "b"),
                           label=kw.get("label"))
                if len(x) == 1
                else ax.plot(x, y, **kw)
            )

        ax = axes[0]
        _p(ax, gen, bf, color="royalblue", lw=2, label="Best fidelity")
        _p(ax, gen, af, color="tomato", lw=1.5, ls="--", alpha=0.8, label="Mean fidelity")
        ax.axhline(1.0, color="green", ls=":", lw=1.2, label="F=1")
        r0 = self.history.get("raw_gen0_best_fidelity")
        if r0:
            ax.axhline(r0, color="grey", ls=":", lw=1.2, label=f"Gen-0 baseline ({r0:.3f})")
        if self.n_ancilla > 0:
            ub = purity_upper_bound(self.rho_target)
            ax.axhline(ub, color="orange", ls="-.", lw=1.2, label=f"Ceiling ({ub:.3f})")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fidelity")
        ax.set_title("Fidelity")
        ax.legend(fontsize=8)
        ax.set_ylim([-0.02, 1.05])
        ax.grid(alpha=0.3)
        if len(gen) > 1:
            ax.set_xlim([gen[0] - 0.5, gen[-1] + 0.5])

        ax = axes[1]
        _p(ax, gen, cl, color="mediumpurple", lw=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("# Gates")
        ax.set_title("Circuit Depth")
        ax.grid(alpha=0.3)
        if len(gen) > 1:
            ax.set_xlim([gen[0] - 0.5, gen[-1] + 0.5])

        ax = axes[2]
        _p(ax, gen, mr, color="darkorange", lw=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mutation Rate")
        ax.set_title("Adaptive Mutation Rate")
        ax.grid(alpha=0.3)
        if len(gen) > 1:
            ax.set_xlim([gen[0] - 0.5, gen[-1] + 0.5])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_density_matrices(self, save_path=None):
        """Six-panel heatmap: target rho, output rho, and element-wise difference."""
        ro, _ = self.get_output_state()
        fid = quantum_fidelity(self.rho_target, ro)
        labs = [f"|{bin(i)[2:].zfill(self.n_qubits)}>" for i in range(2 ** self.n_qubits)]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            f"Density Matrices  |  F={fid:.6f}  |  "
            f"target purity={self.target_purity:.4f}",
            fontsize=11, fontweight="bold",
        )
        for col, (rho, title, vl) in enumerate(
            [(self.rho_target, "Target", (-1, 1)),
             (ro, "Output", (-1, 1)),
             (self.rho_target - ro, "Delta", (None, None))]
        ):
            for row, (data, part) in enumerate(
                [(np.real(rho), "Real"), (np.imag(rho), "Imag")]
            ):
                ax = axes[row][col]
                kw = dict(cmap="RdBu_r", aspect="auto")
                if vl[0] is not None:
                    kw.update(vmin=vl[0], vmax=vl[1])
                im = ax.imshow(data, **kw)
                ax.set_title(f"{title} -- {part}", fontsize=9)
                ax.set_xticks(range(len(labs)))
                ax.set_xticklabels(labs, fontsize=7)
                ax.set_yticks(range(len(labs)))
                ax.set_yticklabels(labs, fontsize=7)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_ga_for_target(
    rho_target,
    n_qubits,
    n_spdc=-1,
    n_ancilla=-1,
    herald_qubits=0,
    herald_outcome=None,
    population_size=80,
    max_generations=300,
    init_circuit_length=8,
    fidelity_target=0.995,
    seed=0,
    verbose=True,
    plot=True,
    save_plots=False,
):
    """
    One-call convenience wrapper: construct, run, report, and optionally plot.
    Returns the trained QuantumOpticsGA instance and the best Individual.
    """
    ga = QuantumOpticsGA(
        rho_target, n_qubits,
        n_spdc=n_spdc, n_ancilla=n_ancilla,
        herald_qubits=herald_qubits, herald_outcome=herald_outcome,
        population_size=population_size, max_generations=max_generations,
        init_circuit_length=init_circuit_length,
        fidelity_target=fidelity_target, seed=seed, verbose=verbose,
    )
    best = ga.evolve()
    ga.print_report()
    if plot:
        ga.plot(save_path="ga_evolution.png" if save_plots else None)
        ga.plot_density_matrices(save_path="ga_dm.png" if save_plots else None)
    return ga, best

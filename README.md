# Autonomous Quantum Photonic Circuit Design via Genetic Algorithm

A fully autonomous system that designs quantum optical experiments from scratch. Given any target multi-qubit quantum state, the algorithm searches a physically motivated component library and evolves a photonic circuit that prepares that state — no gradient descent, no neural network, no human guidance.

---

## What This Does, in Plain Terms

Most approaches to quantum circuit synthesis assume you have a nice gate set (CNOT, Hadamard, T-gate…) and work in the abstract. We do something different: we work directly in the language of a real photonic lab.

The system starts with entangled photon pairs from SPDC sources, routes them through a scaffold of beam splitters and detectors, and then uses a genetic algorithm to evolve a sequence of optical components — wave plates, phase shifters, beam splitters, polarising elements — until the output density matrix matches the target to high fidelity.

The fitness function is the Uhlmann-Jozsa fidelity. That's it. The algorithm figures out the rest.

---

## Repository Structure

```
.
├── generate_state.py     # Module 1 — quantum state generation and noise channels
├── library.py            # Module 2 — optical component library and circuit simulator
├── model.py              # Module 3 — genetic algorithm and experiment narrator
├── main.py               # CLI driver script
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run with default settings (4-qubit random pure state, 200 generations)
python main.py

# 3-qubit mixed state with depolarising noise
python main.py --n-qubits 3 --noise-prob 0.15 --noise-type depolarising

# Supply your own target density matrix (if you don't, random target state will be used !)
python main.py --rho-file my_target.npy

# Save all plots
python main.py --save-plots

# Full parameter control
python main.py --n-qubits 4 --population-size 120 --max-generations 500 --fidelity-target 0.999
```

---

## The Component Library (`library.py`)

All components are implemented as exact unitary matrices acting on polarisation-encoded qubits. `|H⟩ ↔ |0⟩`, `|V⟩ ↔ |1⟩`.

### Single-Qubit Components

**Phase Shifter (PS)**

Applies a controlled phase advance to the vertical polarisation component while leaving horizontal unchanged:

$$\text{PS}(\phi) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}, \quad \phi \in [0, 2\pi)$$

Physically realised as a birefringent crystal or electro-optic modulator. The parameter $\phi$ is optimised continuously by the GA.

**Half-Wave Plate (HWP)**

Introduces $\pi$ radians of retardation. Rotates the polarisation state by twice the fast-axis angle $\theta$:

$$\text{HWP}(\theta) = \begin{pmatrix} \cos 2\theta & \sin 2\theta \\ \sin 2\theta & -\cos 2\theta \end{pmatrix}, \quad \theta \in [0, \pi/2)$$

At $\theta = 45°$ this acts as a Pauli-X (bit-flip): $|H\rangle \leftrightarrow |V\rangle$. At $\theta = 22.5°$ it produces the Hadamard transformation.

**Quarter-Wave Plate (QWP)**

Introduces $\pi/2$ retardation, converting between linear and elliptical (or circular) polarisation:

$$\text{QWP}(\theta) = e^{i\pi/4} \begin{pmatrix} \cos^2\theta + i\sin^2\theta & (1-i)\sin\theta\cos\theta \\ (1-i)\sin\theta\cos\theta & \sin^2\theta + i\cos^2\theta \end{pmatrix}, \quad \theta \in [0, \pi/2)$$

At $\theta = 45°$: converts horizontal linear polarisation to right-hand circular polarisation ($|H\rangle \to (|H\rangle - i|V\rangle)/\sqrt{2}$).

**Universal Single-Qubit Rotation (SU2)**

Any $2 \times 2$ unitary can be decomposed via ZYZ Euler angles. Physically implemented as a QWP–HWP–QWP sequence on a beam path:

$$U(\alpha, \beta, \gamma) = R_z(\alpha)\, R_y(\beta)\, R_z(\gamma)$$

where

$$R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}, \qquad R_y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

Parameters: $\alpha \in [0, 2\pi)$, $\beta \in [0, \pi)$, $\gamma \in [0, 2\pi)$.

### Two-Qubit Components

All two-qubit gates act on ordered pairs of qubits and are embedded into the full $2^n$-dimensional Hilbert space via `_embed_two`, which correctly handles non-adjacent qubit pairs using explicit bit-indexing.

**Symmetric 50:50 Beam Splitter (BS)**

The key element of linear-optic quantum computing. Implements Hong-Ou-Mandel interference when both input ports are simultaneously occupied by indistinguishable photons:

$$\text{BS} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 0 & 0 & i \\ 0 & 1 & i & 0 \\ 0 & i & 1 & 0 \\ i & 0 & 0 & 1 \end{pmatrix}$$

Transmission amplitude $1/\sqrt{2}$; reflection amplitude $i/\sqrt{2}$ (the extra $i$ phase is physical — it comes from the phase shift on reflection at the denser medium). No free parameters; entirely fixed by physics.

**Polarising Beam Splitter (PBS)**

Transmits horizontal photons ($|H\rangle$) and reflects vertical photons ($|V\rangle$). Acts as a partial SWAP on the polarisation degree of freedom across two spatial modes:

$$\text{PBS} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

The cornerstone of linear-optic Bell-state measurement and entanglement-swapping protocols.

**CNOT Gate**

Flips the target qubit if and only if the control is $|1\rangle$ (vertically polarised):

$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

In linear optics, CNOT requires ancilla photons and KLM-style post-selection. Treated here as an ideal unitary; the post-selection overhead is absorbed into the heralding layer.

**Controlled-Z Gate (CZ)**

Applies a $-1$ phase to the $|11\rangle$ component only. Symmetric — either qubit may be the control:

$$\text{CZ} = \text{diag}(1, 1, 1, -1)$$

**SWAP Gate**

Exchanges the complete quantum states of two modes: $|ab\rangle \mapsto |ba\rangle$. Perfectly reversible, implementable physically as a crossed optical path or as three CNOT gates:

$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

### SPDC Sources and Scaffolding

Each SPDC source produces one ideal photon pair in the Bell state:

$$|\Phi^+\rangle = \frac{|HH\rangle + |VV\rangle}{\sqrt{2}}$$

The number of sources is set automatically: $n_\text{SPDC} = \min(3,\, \lceil n_\text{qubits}/2 \rceil)$.

Before the GA circuit starts, a deterministic **scaffold** fuses the raw SPDC outputs into a rich entangled starting state. The scaffolds are:

- **1 qubit**: Herald one photon of a Bell pair by post-selecting on a threshold detector click, then rotate the surviving photon with an HWP at $67.5°$ to prepare $|{-}\rangle = (|H\rangle - |V\rangle)/\sqrt{2}$.
- **2 qubits**: One SPDC source directly yields $|\Phi^+\rangle$ — maximally entangled, no heralding needed.
- **3 qubits**: Two SPDC sources; their inner modes are interfered on a BS (Hong-Ou-Mandel); one output port is heralded, leaving a three-qubit W-class entangled state across the remaining modes.
- **4 qubits**: Two SPDC sources; inner modes interfered on a BS (no heralding needed); output is a four-qubit cluster-like state with mutual entanglement across all four modes.

---

## The Algorithm (`model.py`)

### Why Genetic Algorithms?

Quantum circuit synthesis is a search problem with a combinatorial-plus-continuous structure: discrete choices (which gate? which qubits?) entangled with continuous parameters (which angle?). Gradient-based methods struggle here because the search space has discontinuities wherever gate types or qubit assignments change. Reinforcement learning approaches tend to require enormous sample counts. Genetic algorithms handle this naturally: they explore the discrete structure globally while continuous parameter optimisation handles the fine-tuning locally.

### Representation

Each individual in the population is a variable-length list of `GateOp` descriptors. A `GateOp` stores the gate name, the target qubit(s), and any continuous parameters. The circuit is evaluated by composing all gates into a single unitary $U_\text{total}$, evolving the scaffold state $\rho_\text{scaffold}$ as:

$$\rho_\text{out} = U_\text{total}\, \rho_\text{scaffold}\, U_\text{total}^\dagger$$

then optionally tracing out ancilla qubits and applying heralding post-selection.

### Fitness Function

The fitness of a circuit is the Uhlmann-Jozsa fidelity between the produced state $\rho_\text{out}$ and the target $\rho_\text{target}$:

$$F(\rho_\text{target}, \rho_\text{out}) = \left(\text{Tr}\sqrt{\sqrt{\rho_\text{target}}\,\rho_\text{out}\,\sqrt{\rho_\text{target}}}\right)^2$$

This is computed via eigendecomposition: $\rho_\text{target} = V \Lambda V^\dagger$, then $\sqrt{\rho_\text{target}} = V \sqrt{\Lambda} V^\dagger$, then the eigenvalues of $\sqrt{\rho_\text{target}}\,\rho_\text{out}\,\sqrt{\rho_\text{target}}$ are computed and their square roots summed. The result is clipped to $[0, 1]$ for numerical stability.

For mixed targets, a fidelity ceiling is computed as the largest eigenvalue of $\rho_\text{target}$ (the maximum fidelity any pure state can achieve against it), and the GA's convergence threshold is set accordingly.

### Genetic Operators

**Selection**: Tournament selection with configurable tournament size (default 5). The top $k$ individuals (default $k = 6$) are carried forward unchanged as elites.

**Crossover**: Uniform crossover with rate 0.75. For each gate slot up to the length of the shorter parent, the gate is independently assigned from parent 1 or parent 2 with probability 0.5. Tail gates from the longer parent are appended unchanged.

**Mutation**: Five stochastic operators applied per individual per generation:
1. Gate replacement — swap a gate for a freshly sampled random gate (rate $0.4 \cdot \mu$)
2. Parameter perturbation — add Gaussian noise $\mathcal{N}(0, \sigma)$ to a continuous parameter, wrapping within bounds (rate $0.6 \cdot \mu$)
3. Gate insertion — insert a new random gate at a random position (rate $0.5 \cdot \mu$, only if circuit is below max length)
4. Gate deletion — remove a random gate (rate $0.4 \cdot \mu$, only if circuit has more than one gate)
5. Qubit reassignment — reassign the qubit target(s) of a random gate (rate $0.3 \cdot \mu$)

### Adaptive Mutation Rate

The mutation rate $\mu$ is not fixed. Every generation, the algorithm checks for stagnation: if the best fidelity has changed by less than $3 \times 10^{-4}$ over the last 25 generations, $\mu$ is increased by a factor of 1.15 (up to a maximum of 0.75). Otherwise $\mu$ decays gently by a factor of 0.97 back towards the base rate. This drives exploration when the population is stuck and exploitation when progress is being made.

### Hybrid Local Optimisation

Every 15 generations, the top $k$ elites undergo **coordinate-ascent parameter optimisation**. For each continuous parameter in each gate, the algorithm runs a **golden-section search** over the parameter's valid range — finding the local maximum of fidelity as a function of that one angle while holding all others fixed. The golden-section method converges in $O(\log(1/\varepsilon))$ iterations with no gradient required.

At the end of evolution, the global best individual undergoes a final intensive local optimisation pass (5 restarts, 80 coordinate-ascent iterations).

This hybrid design is the core of the approach: the GA's global structure search finds good circuit topologies; the local optimiser then precisely tunes every angle within that topology. Neither component alone would achieve high fidelity quickly.

### Evolution Loop Summary

```
Initialise population with random circuits of length L₀
Evaluate all individuals (compute fidelity for each)

For each generation:
    Sort population by fitness
    Every 15 generations: run golden-section local optimisation on top-k elites
    Record history (fidelity, circuit depth, mutation rate, parameter trajectory)
    If best fidelity ≥ target: stop early
    Update adaptive mutation rate
    Build next generation:
        Carry forward top-k elites unchanged
        Fill remaining slots via tournament selection → crossover → mutation
    Evaluate new individuals

Final pass: intensive local optimisation on global best
```

### Mixed State Handling

When the target has purity $< 1$, the system automatically allocates ancilla qubits (equal in number to the system qubits). The circuit acts on the larger system + ancilla space, and the ancilla qubits are traced out at the end to produce a mixed output state. This is the standard purification approach: any mixed state $\rho$ on $n$ qubits can be obtained as the partial trace of a pure state on $2n$ qubits.

---

## State Generation (`generate_state.py`)

Target states are generated as **Haar-random** pure states: a complex Gaussian vector is sampled and normalised. The Haar measure is the unique unitarily-invariant distribution on the complex unit sphere — it is the natural notion of "uniformly random" over quantum states and avoids any bias toward special structured states.

Three noise channels are available to produce mixed targets:

**Depolarising**: $\rho \to (1-p)\rho + \frac{p}{d}I$. With probability $p$ the state is replaced by the maximally mixed state. Isotropic — no preferred direction.

**Dephasing**: $\rho_{ij} \to (1-p)\rho_{ij}$ for $i \neq j$; diagonal unchanged. Destroys quantum coherences (off-diagonal elements) while preserving populations. Models pure phase noise.

**Amplitude Damping**: Modelled via Kraus operators applied independently per qubit:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \qquad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$
Models spontaneous decay $|1\rangle \to |0\rangle$ or photon loss. Asymmetric — preferentially damps the excited state.

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--n-qubits` | 4 | Number of system qubits (1–4) |
| `--noise-prob` | 0.0 | Noise strength in [0, 1] |
| `--noise-type` | depolarising | depolarising / dephasing / amplitude_damping |
| `--train-seed` | 7 | RNG seed for the target state |
| `--n-test` | 8 | Number of held-out test states |
| `--test-seed` | 9999 | RNG seed for held-out states |
| `--population-size` | 80 | GA population size |
| `--max-generations` | 200 | Maximum number of generations |
| `--init-circuit-len` | 10 | Initial circuit length |
| `--fidelity-target` | 0.995 | Early-stop threshold |
| `--ga-seed` | 42 | RNG seed for the GA |
| `--save-plots` | — | Save plots as PNG |
| `--no-plots` | — | Disable all plotting |
| `--rho-file` | None | Path to `.npy` density matrix file |

---

## Outputs

The script produces three diagnostic plots:

1. **Evolution plot** — fidelity history (best and mean), circuit depth over generations, adaptive mutation rate trajectory.
2. **Density matrix plot** — six-panel heatmap comparing the real and imaginary parts of $\rho_\text{target}$, $\rho_\text{out}$, and the element-wise difference $\Delta\rho$.
3. **Parameter trajectory plot** (via `ga.plot_param_trajectories()`) — how each gate's angles evolved across generations.

A full human-readable description of the final experiment (scaffold steps + gate-by-gate narration with physical interpretation) can be generated via `ga.describe_experiment()`.

---

## Judgement Criteria Mapping

| Criterion | How we address it |
|---|---|
| Beauty of the algorithm | Hybrid GA + golden-section coordinate ascent; scaffold uses real photonic physics (HOM interference, SPDC, heralding); adaptive mutation with stagnation detection |
| Reproducibility | Fully seeded (`--ga-seed`, `--train-seed`, `--test-seed`); no external data dependencies; `requirements.txt` pins exact versions |
| Algorithm runtime | Early stopping when fidelity target is reached; local opt only on top-k elites every 15 generations; golden-section convergence in $O(\log 1/\varepsilon)$ |
| Autonomous design | Zero human guidance at inference time; given any density matrix, the system produces a complete experimental setup description |

---

## Dependencies

```
numpy>=1.24.0
matplotlib>=3.7.0
```

No quantum computing frameworks required. The entire simulation is implemented from scratch using dense matrix arithmetic.

---

## Supplying a Custom Target

```python
import numpy as np

# Construct any valid density matrix
rho = ...   # complex numpy array, shape (2^n, 2^n)
            # must be Hermitian, trace-1, positive semi-definite
np.save("my_target.npy", rho)
```

```bash
python main.py --rho-file my_target.npy
```

The script validates Hermiticity, unit trace, and positive semi-definiteness before passing the matrix to the GA.


import argparse
import sys

import numpy as np

from generate_state import generate_dataset
from model import QuantumOpticsGA


# ---------------------------------------------------------------------------
# CLI definition ( makiing it so that you can change parameters from cli only)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Genetic algorithm for quantum photonic circuit design.\n"
            "All flags are optional; defaults match the original CFG dict."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --n-qubits 3 --noise-prob 0.1 --save-plots\n"
            "  python main.py --rho-file my_target.npy\n"
            "  python main.py --max-generations 500 --population-size 120 --fidelity-target 0.999\n"
        ),
    )

    # ── State generation ──────────────────────────────────────────────────
    gen = p.add_argument_group("State generation (ignored when --rho-file is set)")
    gen.add_argument(
        "--n-qubits", type=int, default=4, metavar="INT",
        help="Number of system qubits: 2, 3, or 4 (default: 4)",
    )
    gen.add_argument(
        "--noise-prob", type=float, default=0.0, metavar="FLOAT",
        help="Noise strength in [0.0, 1.0]; 0.0 = pure state (default: 0.0)",
    )
    gen.add_argument(
        "--noise-type", type=str, default="depolarising",
        choices=["depolarising", "dephasing", "amplitude_damping"],
        metavar="TYPE",
        help="Noise model: depolarising | dephasing | amplitude_damping (default: depolarising)",
    )
    gen.add_argument(
        "--train-seed", type=int, default=7, metavar="INT",
        help="RNG seed for the training target state (default: 7)",
    )

    # ── Held-out test set ─────────────────────────────────────────────────
    tst = p.add_argument_group("Held-out evaluation")
    tst.add_argument(
        "--n-test", type=int, default=8, metavar="INT",
        help="Number of held-out test states (default: 8)",
    )
    tst.add_argument(
        "--test-seed", type=int, default=9999, metavar="INT",
        help="RNG seed for the held-out test states (default: 9999)",
    )

    # ── GA hyper-parameters ───────────────────────────────────────────────
    ga = p.add_argument_group("GA hyper-parameters")
    ga.add_argument(
        "--population-size", type=int, default=80, metavar="INT",
        help="GA population size (default: 80)",
    )
    ga.add_argument(
        "--max-generations", type=int, default=200, metavar="INT",
        help="Maximum number of GA generations (default: 200)",
    )
    ga.add_argument(
        "--init-circuit-len", type=int, default=10, metavar="INT",
        help="Initial circuit length (default: 10)",
    )
    ga.add_argument(
        "--fidelity-target", type=float, default=0.995, metavar="FLOAT",
        help="Fidelity threshold to stop evolution early (default: 0.995)",
    )
    ga.add_argument(
        "--ga-seed", type=int, default=42, metavar="INT",
        help="RNG seed for the GA (default: 42)",
    )

    # ── Output / plotting ─────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument(
        "--save-plots", action="store_true",
        help="Save evolution and density-matrix plots as PNG files",
    )
    out.add_argument(
        "--no-plots", action="store_true",
        help="Disable all plotting (useful in headless / server environments)",
    )

    # ── Custom target density matrix ──────────────────────────────────────
    p.add_argument(
        "--rho-file", type=str, default=None, metavar="PATH",
        help=(
            "Path to a .npy file containing a custom target density matrix "
            "(complex numpy array of shape (2^n, 2^n)). "
            "n_qubits is inferred from the matrix shape automatically. "
            "--noise-prob / --noise-type / --train-seed are ignored. "
            "Save your matrix with: np.save('my_target.npy', rho)"
        ),
    )

    return p


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_density_matrix(rho: np.ndarray, label: str = "rho") -> int:
    """
    Check that rho is a valid square density matrix and return n_qubits.
    Raises SystemExit with a clear message on any violation.
    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        sys.exit(
            f"[ERROR] {label} must be a square 2-D array; "
            f"got shape {rho.shape}."
        )

    dim = rho.shape[0]
    n = dim.bit_length() - 1          # log2(dim) for exact powers of two
    if 2 ** n != dim:
        sys.exit(
            f"[ERROR] {label} dimension {dim} is not a power of two. "
            f"Expected 2^n for some integer n."
        )
    if n < 1 or n > 4:
        sys.exit(
            f"[ERROR] {label} implies n_qubits = {n}. "
            f"This GA supports 1–4 qubits."
        )

    # Hermiticity
    if not np.allclose(rho, rho.conj().T, atol=1e-6):
        sys.exit(f"[ERROR] {label} is not Hermitian (rho != rho†).")

    # Trace ≈ 1
    tr = np.real(np.trace(rho))
    if not np.isclose(tr, 1.0, atol=1e-5):
        sys.exit(
            f"[ERROR] {label} has trace {tr:.6f}; expected 1.0. "
            f"Please renormalise your matrix before passing it in."
        )

    # Positive semi-definiteness
    eigvals = np.linalg.eigvalsh(rho)
    if np.any(eigvals < -1e-6):
        sys.exit(
            f"[ERROR] {label} has negative eigenvalue(s) "
            f"(min = {eigvals.min():.2e}). "
            f"It is not a valid density matrix."
        )

    return n


def load_rho_file(path: str):
    """Load and validate a .npy density matrix file. Returns (rho, n_qubits)."""
    try:
        rho = np.load(path, allow_pickle=False)
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: '{path}'")
    except Exception as exc:
        sys.exit(f"[ERROR] Could not load '{path}': {exc}")

    rho = rho.astype(complex)
    n_qubits = _validate_density_matrix(rho, label=f"'{path}'")
    return rho, n_qubits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Resolve target density matrix ─────────────────────────────────────
    if args.rho_file is not None:
        print(f"\n  [--rho-file] Loading target density matrix from '{args.rho_file}' ...")
        rho_target, n_qubits = load_rho_file(args.rho_file)
        purity = float(np.real(np.trace(rho_target @ rho_target)))
        print(f"  Loaded: shape={rho_target.shape}  n_qubits={n_qubits}  purity={purity:.6f}")
        print(f"  (--noise-prob, --noise-type, --train-seed are ignored)\n")
    else:
        n_qubits = args.n_qubits
        print(
            f"\n  Generating random target state  "
            f"(n_qubits={n_qubits}, noise_prob={args.noise_prob}, "
            f"noise_type={args.noise_type}, seed={args.train_seed}) ..."
        )
        ds = generate_dataset(
            1,
            n_qubits   = n_qubits,
            noise_prob = args.noise_prob,
            noise_type = args.noise_type,
            seed       = args.train_seed,
        )
        rho_target = ds[0]["rho"]

    # ── Run GA ────────────────────────────────────────────────────────────
    ga = QuantumOpticsGA(
        rho_target          = rho_target,
        n_qubits            = n_qubits,
        population_size     = args.population_size,
        max_generations     = args.max_generations,
        init_circuit_length = args.init_circuit_len,
        fidelity_target     = args.fidelity_target,
        seed                = args.ga_seed,
        verbose             = True,
    )
    ga.evolve()
    ga.print_report()

    # ── Held-out evaluation ───────────────────────────────────────────────
    if args.rho_file is not None:
        # No natural test distribution for a custom matrix; evaluate against itself.
        print("\n  [held-out] --rho-file mode: evaluating circuit against the supplied target.")
        ga.evaluate_on_test_set([rho_target])
    else:
        test_ds = generate_dataset(
            args.n_test,
            n_qubits   = n_qubits,
            noise_prob = args.noise_prob,
            noise_type = args.noise_type,
            seed       = args.test_seed,
        )
        ga.evaluate_on_test_set([s["rho"] for s in test_ds])

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        sp = args.save_plots
        ga.plot(save_path="ga_evolution.png" if sp else None)
        ga.plot_density_matrices(save_path="ga_density_matrices.png" if sp else None)
    else:
        print("\n  [--no-plots] Plotting skipped.")

    # ── Summary line ──────────────────────────────────────────────────────
    r0 = ga.history["raw_gen0_best_fidelity"]
    print(
        f"\n  baseline={r0:.4f}  ->  final={ga.best.fitness:.4f}  "
        f"(+{ga.best.fitness - r0:.4f})  "
        f"spdc={ga.n_spdc}  ancilla={ga.n_ancilla}"
    )


if __name__ == "__main__":
    main()
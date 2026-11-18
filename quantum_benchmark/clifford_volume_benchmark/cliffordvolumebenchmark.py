"""Clifford Volume (CLV) benchmark, Sec. 2.2 and Appendix A of arXiv:2512.19413.

For a given n, K random n-qubit Clifford unitaries C_k are sampled. For each
of them n_m = 4 stabilizer generators S_i^k = C Z_i C^dag (ideal <S> = 1) and
n_m = 4 destabilizers D_i^k = C X_i C^dag (ideal <D> = 0) are drawn at random,
and one circuit |0^n> -> C_k -> basis change -> Z measurement is emitted per
Pauli operator. The n-qubit step passes if, for every k and i,

    (5)  <S_i^k> - 2 sigma >= 1/e        and   |<D_i^k>| + 2 sigma <= 1/(2e)
    (7)  mean<S^k> - 5 sigma_bar >= 1/e  and   |mean<D^k>| + 5 sigma_bar <= 1/(2e)

with sigma = sqrt((1 - <P>^2) / L) (Eq. 6) and
sigma_bar = (1/n_m) sqrt(sum_i sigma_i^2) (Eq. 8).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import stim  # type: ignore

from ..utils.quantum_circuit import QuantumCircuit
from ..utils.volumetric_benchmark import Benchmark

TAU_S = 1.0 / np.e
TAU_D = 1.0 / (2.0 * np.e)
N_MEASUREMENTS = 4  # n_m in the paper


class CliffordVolumeBenchmark(Benchmark):
    """Clifford Volume benchmark for a fixed number of qubits n.

    Args:
      number_of_qubits: n.
      sample_size: K, number of random Clifford unitaries (paper: K = 4).
      **kwargs: Forwarded to :class:`Benchmark` (``shots``, ``format``,
        ``target_sdk``, ``seed``, ``auto_save``, ...).
    """

    BENCHMARK_NAME: str = "clifford"

    def __init__(self, number_of_qubits: int, sample_size: int = 4, **kwargs: Any):
        super().__init__(number_of_qubits=number_of_qubits, sample_size=sample_size, **kwargs)
        self.number_of_measurements: int = self._compute_number_of_measurements()

    def _compute_number_of_measurements(self) -> int:
        # n_m = 4 stabilizers and 4 destabilizers; only n exist for n < 4.
        return min(self.number_of_qubits, N_MEASUREMENTS)

    # ------------------------------------------------------------------
    # Circuit generation
    # ------------------------------------------------------------------
    def _stim_to_quantum_circuit(self, stim_circuit: stim.Circuit) -> QuantumCircuit:
        """Convert a stim circuit over {H, S, CX} into a :class:`QuantumCircuit`."""
        n = self.number_of_qubits
        qc = QuantumCircuit(number_of_qubits=n, number_of_classical_bits=n)
        for instr in stim_circuit:
            name = instr.name.upper()
            qubits = [t.value for t in instr.targets_copy()]
            if name == "H":
                for q in qubits:
                    qc.add_h_gate(q)
            elif name == "S":
                for q in qubits:
                    qc.add_s_gate(q)
            elif name in ("CX", "CNOT"):
                for c, t in zip(qubits[0::2], qubits[1::2]):
                    qc.add_cx_gate(c, t)
            elif name == "TICK":
                continue
            else:
                raise ValueError(f"Unsupported gate from stim: {name}")
        return qc

    def _measurement_circuit(self, base: QuantumCircuit, pauli: str) -> QuantumCircuit:
        """Append the single-qubit basis change for ``pauli`` and measure all qubits.

        X -> H, Y -> Sdg then H, Z/I -> nothing.
        """
        qc = deepcopy(base)
        for q, op in enumerate(pauli.lstrip("+-")):
            if op == "X":
                qc.add_h_gate(q)
            elif op == "Y":
                qc.add_sdg_gate(q)
                qc.add_h_gate(q)
        for q in range(self.number_of_qubits):
            qc.add_measurement(q, q)
        return qc

    def _create_single_sample(self, sample_id: int) -> Dict[str, Any]:
        n = self.number_of_qubits
        m = self.number_of_measurements

        tableau = stim.Tableau.random(n)
        base_qc = self._stim_to_quantum_circuit(tableau.to_circuit(method="elimination"))

        # Signed Pauli strings, qubit 0 first, e.g. "+XZI".
        stabilizers = [str(tableau.z_output(q)).replace("_", "I") for q in range(n)]
        destabilizers = [str(tableau.x_output(q)).replace("_", "I") for q in range(n)]

        stab_idx = self.rng.choice(n, size=m, replace=False)
        dest_idx = self.rng.choice(n, size=m, replace=False)

        circuits: List[Dict[str, Any]] = []
        for kind, paulis, idx in (
            ("stabilizer", stabilizers, stab_idx),
            ("destabilizer", destabilizers, dest_idx),
        ):
            tag = "stab" if kind == "stabilizer" else "destab"
            for i, q in enumerate(idx):
                pauli = paulis[int(q)]
                circuits.append(
                    {
                        "circuit_id": f"{sample_id}_{tag}_{i}",
                        "observable": pauli,
                        "qasm": self._measurement_circuit(base_qc, pauli).to_qasm(
                            self.emitter_options
                        ),
                        "metadata": {"kind": kind, "index": i, "generator": int(q)},
                    }
                )

        return {
            "sample_id": sample_id,
            "sample_metadata": {
                "type": "clifford",
                "number_of_measurements": m,
                "clifford_operator": {
                    "stabilizers": stabilizers,
                    "destabilizers": destabilizers,
                },
            },
            "circuits": circuits,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_benchmark(
        self,
        *,
        verbose: bool = True,
        auto_save: Optional[bool] = None,
        save_to: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Apply the CLV acceptance criteria, Eqs. (5)-(8).

        Returns:
          dict with ``passed`` (all K unitaries pass), ``per_sample`` details
          and the thresholds.
        """
        if self.samples is None or self.experimental_results is None:
            raise ValueError("Generate/load the benchmark and add experimental results first.")
        if auto_save is not None:
            self.auto_save = bool(auto_save)

        per_sample: Dict[int, Dict[str, Any]] = {}
        for sample in self.samples:
            sid = sample["sample_id"]
            data: Dict[str, List[tuple]] = {"stabilizer": [], "destabilizer": []}
            for c in sample["circuits"]:
                kind = c.get("metadata", {}).get("kind")
                if kind in data:
                    ev, sig, _ = self._ev_and_sigma(c["circuit_id"], c["observable"])
                    data[kind].append((c["observable"], ev, sig))

            if not data["stabilizer"] or not data["destabilizer"]:
                raise ValueError(f"Sample {sid} has no stabilizer or destabilizer circuits.")

            s_ev = np.array([e for _, e, _ in data["stabilizer"]])
            s_sig = np.array([s for _, _, s in data["stabilizer"]])
            d_ev = np.array([e for _, e, _ in data["destabilizer"]])
            d_sig = np.array([s for _, _, s in data["destabilizer"]])

            # Eq. (5): worst case over single operators.
            min_stab = float(np.min(s_ev - 2 * s_sig))
            max_dest = float(np.max(np.abs(d_ev) + 2 * d_sig))
            crit_I = min_stab >= TAU_S and max_dest <= TAU_D

            # Eqs. (7), (8): averages with sigma_bar = sqrt(sum sigma^2) / n_m.
            mean_s, mean_d = float(np.mean(s_ev)), float(np.mean(d_ev))
            sbar_s = float(np.sqrt(np.sum(s_sig**2)) / len(s_sig))
            sbar_d = float(np.sqrt(np.sum(d_sig**2)) / len(d_sig))
            avg_stab = mean_s - 5 * sbar_s
            avg_dest = abs(mean_d) + 5 * sbar_d
            crit_II = avg_stab >= TAU_S and avg_dest <= TAU_D

            per_sample[sid] = {
                "passed": bool(crit_I and crit_II),
                "stabilizers": data["stabilizer"],
                "destabilizers": data["destabilizer"],
                "criterion_I": {
                    "passed": bool(crit_I),
                    "min_stab_minus_2sigma": min_stab,
                    "max_abs_dest_plus_2sigma": max_dest,
                },
                "criterion_II": {
                    "passed": bool(crit_II),
                    "mean_stab": mean_s,
                    "mean_dest": mean_d,
                    "sigma_bar_stab": sbar_s,
                    "sigma_bar_dest": sbar_d,
                    "mean_stab_minus_5sigma": avg_stab,
                    "abs_mean_dest_plus_5sigma": avg_dest,
                },
            }

        passed = all(v["passed"] for v in per_sample.values())
        out = {
            "number_of_qubits": self.number_of_qubits,
            "passed": bool(passed),
            "per_sample": per_sample,
            "thresholds": {"tau_S": TAU_S, "tau_D": TAU_D},
        }
        self.experimental_results["evaluation"] = out

        if verbose:
            print(f"CLV n={self.number_of_qubits}: thresholds 1/e={TAU_S:.4f}, 1/(2e)={TAU_D:.4f}")
            for sid, v in per_sample.items():
                c1, c2 = v["criterion_I"], v["criterion_II"]
                print(
                    f"  C_{sid}: min(<S>-2s)={c1['min_stab_minus_2sigma']:.4f}  "
                    f"max(|<D>|+2s)={c1['max_abs_dest_plus_2sigma']:.4f}  "
                    f"mean<S>-5s={c2['mean_stab_minus_5sigma']:.4f}  "
                    f"|mean<D>|+5s={c2['abs_mean_dest_plus_5sigma']:.4f}  "
                    f"-> {'PASS' if v['passed'] else 'FAIL'}"
                )
            print(f"  benchmark {'PASSED' if passed else 'FAILED'} for n={self.number_of_qubits}")

        self._maybe_save(save_to)
        return out

    def plot_expectation_values(self, ax=None):
        """Plot <S_i^k> and <D_i^k> with 2 sigma error bars and the thresholds (cf. Fig. 12)."""
        import matplotlib.pyplot as plt

        if self.experimental_results is None or "evaluation" not in self.experimental_results:
            raise ValueError("Run evaluate_benchmark() first.")
        per_sample = self.experimental_results["evaluation"]["per_sample"]
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        K = len(per_sample)
        width = 0.8 / max(K, 1)
        for k, (sid, v) in enumerate(per_sample.items()):
            for kind, color in (("stabilizers", "tab:blue"), ("destabilizers", "tab:red")):
                ev = np.array([e for _, e, _ in v[kind]])
                sg = np.array([s for _, _, s in v[kind]])
                x = 1 + np.arange(len(ev)) + (k - (K - 1) / 2) * width
                ax.errorbar(x, ev, yerr=2 * sg, fmt="o", color=color, capsize=3,
                            label=kind.capitalize() if k == 0 else None)
        for y, c in ((TAU_S, "tab:blue"), (TAU_D, "tab:red"), (-TAU_D, "tab:red")):
            ax.axhline(y, ls="--", color=c, lw=1)
        ax.set_xlabel("Operator index")
        ax.set_ylabel("Expectation value")
        ax.set_title(f"CLV, n = {self.number_of_qubits}")
        ax.legend()
        return ax

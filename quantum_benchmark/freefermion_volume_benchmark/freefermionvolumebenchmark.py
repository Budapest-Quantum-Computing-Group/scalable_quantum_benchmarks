"""Free-Fermion Volume (FFV) benchmark, Sec. 2.3 and Appendix B of arXiv:2512.19413.

For a given n, K Haar-random O in SO(2n) are sampled. For each of them:

* a random Majorana index i is chosen and the state rho_i with <m_j> = delta_ij
  is prepared (qubit i//2 in |+> for even i, in |+i> for odd i, others |0>);
* the free-fermion unitary F(O) is applied, compiled from the Givens-rotation
  decomposition O = S * prod G (Eqs. 31-39); then <m_k> = O_ki (Eq. 21);
* the N(n) Majorana operators with the largest |O_ki| are measured (set J),
  N(n) = 2n for n <= 10 and 20 + floor(n/5) otherwise (Eq. 23);
* a second index j != i is drawn for the "orthogonal" combination.

Majorana operators use the Jordan-Wigner map with 0-based indices
m_{2q} = Z_0...Z_{q-1} X_q,  m_{2q+1} = Z_0...Z_{q-1} Y_q.

The n-qubit step passes if, for every sampled O^k (Eqs. 26-28),

    R_ii - 2 sigma_par  >= 1/e,      R_ii = sum_J O_ki <m_k> / sum_J O_ki^2
    |R_ij| + 2 sigma_perp <= 1/(2e), R_ij = sum_J O_kj <m_k>
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.stats import special_ortho_group

from ..utils.quantum_circuit import QuantumCircuit
from ..utils.so_decomposition import so_decomposition
from ..utils.volumetric_benchmark import Benchmark

TAU_PAR = 1.0 / np.e
TAU_PERP = 1.0 / (2.0 * np.e)


def majorana_pauli(j: int, n: int) -> str:
    """Jordan-Wigner Pauli string (qubit 0 first) of the 0-based Majorana m_j."""
    q = j // 2
    return "Z" * q + ("X" if j % 2 == 0 else "Y") + "I" * (n - q - 1)


def number_of_measured_majoranas(n: int) -> int:
    """N(n) of Eq. (23): 2n for n <= 10, else 20 + floor(n/5)."""
    return 2 * n if n <= 10 else 20 + n // 5


class FreeFermionVolumeBenchmark(Benchmark):
    """Free-Fermion Volume benchmark for a fixed number of qubits n.

    Args:
      number_of_qubits: n.
      sample_size: K, number of random SO(2n) matrices (paper: K = 4).
      **kwargs: Forwarded to :class:`Benchmark` (``shots``, ``format``,
        ``target_sdk``, ``seed``, ``auto_save``, ...).
    """

    BENCHMARK_NAME: str = "free_fermion"

    def __init__(self, number_of_qubits: int, sample_size: int = 4, **kwargs: Any):
        super().__init__(number_of_qubits=number_of_qubits, sample_size=sample_size, **kwargs)
        self.number_of_measurements: int = self._compute_number_of_measurements()

    def _compute_number_of_measurements(self) -> int:
        return number_of_measured_majoranas(self.number_of_qubits)

    # ------------------------------------------------------------------
    # Circuit generation
    # ------------------------------------------------------------------
    def _build_free_fermion_circuit(self, givens_rotations, state_index: int) -> QuantumCircuit:
        """State preparation rho_i followed by the Givens-rotation circuit (Eqs. 36-37)."""
        n = self.number_of_qubits
        qc = QuantumCircuit(number_of_qubits=n, number_of_classical_bits=n)

        q = state_index // 2
        qc.add_h_gate(q)  # |+>: <X_q> = 1  -> <m_{2q}> = 1
        if state_index % 2 == 1:
            qc.add_s_gate(q)  # |+i>: <Y_q> = 1 -> <m_{2q+1}> = 1

        for gr in givens_rotations:
            k, theta = gr.n, gr.theta
            if k % 2 == 0:
                # modes (2l, 2l+1): single-qubit Rz on qubit l
                qc.add_rz_gate(k // 2, theta)
            else:
                # modes (2l+1, 2l+2): exp(-i theta X_l X_{l+1} / 2)
                l = (k - 1) // 2
                qc.add_h_gate(l)
                qc.add_h_gate(l + 1)
                qc.add_cx_gate(l, l + 1)
                qc.add_rz_gate(l + 1, theta)
                qc.add_cx_gate(l, l + 1)
                qc.add_h_gate(l)
                qc.add_h_gate(l + 1)
        return qc

    def _diagonal_pauli_layer(self, signs: np.ndarray) -> str:
        """Pauli layer implementing the diagonal S = diag(s_k), Eq. (39): prod_{s_k=-1} m_k."""
        n = self.number_of_qubits
        layer = "I" * n
        for k, s in enumerate(signs):
            if s < 0:
                layer = _pauli_product(layer, majorana_pauli(k, n))
        return layer

    def _create_single_sample(self, sample_id: int) -> Dict[str, Any]:
        n = self.number_of_qubits
        O = special_ortho_group.rvs(2 * n, random_state=self.rng)

        state_index = int(self.rng.integers(2 * n))
        others = [j for j in range(2 * n) if j != state_index]
        orthogonal_index = int(self.rng.choice(others))

        # J: the N(n) Majorana indices with largest |O_ki| (k over rows of column i).
        measured = np.argsort(-np.abs(O[:, state_index]))[: self.number_of_measurements]
        measured = [int(k) for k in measured]

        rotations, D = so_decomposition(O)
        base_qc = self._build_free_fermion_circuit(rotations, state_index)
        for q, p in enumerate(self._diagonal_pauli_layer(np.diag(D))):
            if p == "X":
                base_qc.add_x_gate(q)
            elif p == "Y":
                base_qc.add_y_gate(q)
            elif p == "Z":
                base_qc.add_z_gate(q)

        circuits: List[Dict[str, Any]] = []
        for idx, k in enumerate(measured):
            qc = deepcopy(base_qc)
            q = k // 2
            if k % 2 == 0:
                qc.add_h_gate(q)
            else:
                qc.add_sdg_gate(q)
                qc.add_h_gate(q)
            for qq in range(n):
                qc.add_measurement(qq, qq)
            circuits.append(
                {
                    "circuit_id": f"{sample_id}_maj_{idx}",
                    "observable": majorana_pauli(k, n),
                    "qasm": qc.to_qasm(self.emitter_options),
                    "metadata": {"kind": "majorana", "majorana_index": k},
                }
            )

        return {
            "sample_id": sample_id,
            "sample_metadata": {
                "type": "free_fermion",
                "orthogonal_matrix": O,
                "initial_state_index": state_index,
                "orthogonal_index": orthogonal_index,
                "measured_majorana_indices": measured,
                "number_of_measurements": self.number_of_measurements,
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
        """Apply the FFV acceptance criteria, Eqs. (26)-(28).

        Returns:
          dict with ``passed`` (all K samples pass), ``per_sample`` details
          (R_ii, R_ij, sigmas) and the thresholds.
        """
        if self.samples is None or self.experimental_results is None:
            raise ValueError("Generate/load the benchmark and add experimental results first.")
        if auto_save is not None:
            self.auto_save = bool(auto_save)

        per_sample: Dict[int, Dict[str, Any]] = {}
        for sample in self.samples:
            sid = sample["sample_id"]
            meta = sample["sample_metadata"]
            O = np.asarray(meta["orthogonal_matrix"], dtype=float)
            i = int(meta["initial_state_index"])
            j = int(meta["orthogonal_index"])

            ks, evs, var = [], [], []
            for c in sample["circuits"]:
                if c.get("metadata", {}).get("kind") != "majorana":
                    continue
                ev, _, L = self._ev_and_sigma(c["circuit_id"], c["observable"])
                ks.append(int(c["metadata"]["majorana_index"]))
                evs.append(ev)
                var.append(max(0.0, 1.0 - ev * ev) / L)
            ks_a, evs_a, var_a = np.array(ks), np.array(evs), np.array(var)

            a = O[ks_a, i]  # O_ki, k in J
            b = O[ks_a, j]  # O_kj, k in J
            norm = float(np.sum(a**2))

            R_ii = float(np.dot(a, evs_a) / norm)
            R_ij = float(np.dot(b, evs_a))
            sigma_par = float(np.sqrt(np.sum(a**2 * var_a)) / norm)
            sigma_perp = float(np.sqrt(np.sum(b**2 * var_a)))

            par = R_ii - 2 * sigma_par
            perp = abs(R_ij) + 2 * sigma_perp
            passed = par >= TAU_PAR and perp <= TAU_PERP

            per_sample[sid] = {
                "passed": bool(passed),
                "R_ii": R_ii,
                "R_ij": R_ij,
                "sigma_parallel": sigma_par,
                "sigma_perpendicular": sigma_perp,
                "R_ii_minus_2sigma": par,
                "abs_R_ij_plus_2sigma": perp,
                "sum_J_O_ki_squared": norm,
                "state_index": i,
                "orthogonal_index": j,
            }

        passed = all(v["passed"] for v in per_sample.values())
        out = {
            "number_of_qubits": self.number_of_qubits,
            "passed": bool(passed),
            "per_sample": per_sample,
            "thresholds": {"tau_parallel": TAU_PAR, "tau_perpendicular": TAU_PERP},
        }
        self.experimental_results["evaluation"] = out

        if verbose:
            print(f"FFV n={self.number_of_qubits}: thresholds 1/e={TAU_PAR:.4f}, 1/(2e)={TAU_PERP:.4f}")
            for sid, v in per_sample.items():
                print(
                    f"  O_{sid}: R_ii-2s={v['R_ii_minus_2sigma']:.4f}  "
                    f"|R_ij|+2s={v['abs_R_ij_plus_2sigma']:.4f}  "
                    f"-> {'PASS' if v['passed'] else 'FAIL'}"
                )
            print(f"  benchmark {'PASSED' if passed else 'FAILED'} for n={self.number_of_qubits}")

        self._maybe_save(save_to)
        return out


_PAULI_MULT = {
    ("I", "I"): "I", ("I", "X"): "X", ("I", "Y"): "Y", ("I", "Z"): "Z",
    ("X", "I"): "X", ("X", "X"): "I", ("X", "Y"): "Z", ("X", "Z"): "Y",
    ("Y", "I"): "Y", ("Y", "X"): "Z", ("Y", "Y"): "I", ("Y", "Z"): "X",
    ("Z", "I"): "Z", ("Z", "X"): "Y", ("Z", "Y"): "X", ("Z", "Z"): "I",
}


def _pauli_product(p1: str, p2: str) -> str:
    """Product of two Pauli strings up to a global phase (irrelevant for the circuit)."""
    return "".join(_PAULI_MULT[(a, b)] for a, b in zip(p1, p2))

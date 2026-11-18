# Clifford Volume and Free-Fermion Volume benchmarks

Supplementary code for *Clifford Volume and Free Fermion Volume: Complementary
Scalable Benchmarks for Quantum Computers* (arXiv:2512.19413).

For a fixed number of qubits `n`, the code samples the K = 4 random instances,
exports one OpenQASM circuit per measured observable, and evaluates the
acceptance criteria from measured counts:

- **CLV** (Sec. 2.2, App. A): random n-qubit Cliffords (stim, {H, S, CX}).
  4 stabilizers and 4 destabilizers are measured per instance, using
  Eqs. (5)–(8) with thresholds 1/e and 1/(2e).
- **FFV** (Sec. 2.3, App. B): random O ∈ SO(2n), compiled from Givens
  rotations (Sec. 3.2). N(n) Majorana operators are measured per instance
  (Eq. 23), using Eqs. (26)–(28).

The score is the largest `n` that passes (`volume_score`; for FFV every
`n ≤ n_max` must pass: `contiguous=True`).

## Usage

```bash
pip install -e .
```

```python
from quantum_benchmark import CliffordVolumeBenchmark, FreeFermionVolumeBenchmark, volume_score

# 1. initialise: K = 4 instances, L = 512 shots per circuit
bench = CliffordVolumeBenchmark(number_of_qubits=20, sample_size=4, shots=512, seed=1)
# bench = FreeFermionVolumeBenchmark(number_of_qubits=20, sample_size=4, shots=512, seed=1)
bench.create_benchmark()

# 2. export: QASM circuits (+ ids) for execution, or everything as JSON
circuit_ids = bench.get_all_circuit_ids()
qasm_circuits = bench.get_all_circuits()
bench.save_json("clv_n20.json")

# 3. run the circuits on the device, then attach counts {circuit_id: {bitstring: count}}
#    bitstrings are little-endian (rightmost bit = qubit 0, as in Qiskit)
bench = CliffordVolumeBenchmark.load_json("clv_n20.json")
bench.add_experimental_results(counts, platform="my-device")

# 4. evaluate the acceptance criteria for this n
result = bench.evaluate_benchmark()   # result["passed"], result["per_sample"]
bench.save_json("clv_n20_results.json")

# score over the tested n
score = volume_score({18: True, 19: True, 20: result["passed"]})
```

The QASM dialect can be chosen with `format="qasm2" | "qasm3"` and
`target_sdk="qiskit" | "tket" | "braket"`.

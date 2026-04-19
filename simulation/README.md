# Density Matrix Simulation (Qiskit)

A small simulation project for comparing **GHZ** and **W** states under noise using density matrices.

The code runs two noise pipelines side by side:
- Pure-state density matrix + manually applied Kraus channels
- Circuit execution on Qiskit Aer with a configurable noise model

It then computes metrics (entropy, fidelity, concurrence, monogamy), and can export sweep results to CSV for plotting.

## Project Structure

- `main.py`: CLI entry point and demo workflow (single run + parameter sweep + plotting)
- `runner.py`: orchestration for one run (`run`) and sweeps (`loop_run`)
- `states.py`: GHZ/W circuit builders and state conversion helpers
- `noise.py`: Kraus operators, direct density-matrix noise application, and Aer noise-model builder
- `simulation.py`: Aer simulation helper returning final density matrix
- `metrics.py`: quantum-state metric utilities
- `visualization.py`: matplotlib and plotly plotting utilities
- `datadump.py`: convert run/sweep results to pandas DataFrames and CSV files
- `out/`: generated CSV outputs

## Requirements

- Python 3.10+
- Packages:
  - `qiskit`
  - `qiskit-aer`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `plotly`

Install dependencies:

```bash
pip install qiskit qiskit-aer numpy pandas matplotlib plotly
```

## Quick Start

Run the default workflow:

```bash
python main.py
```

Run with custom settings:

```bash
python main.py --bits 3 --p1 0.01 --p2 0.1 --pairs 0-1,1-2
```

### CLI Options

- `--bits`: number of qubits (default: `3`)
- `--p1`: 1-qubit depolarizing probability (default: `0.01`)
- `--p2`: 2-qubit depolarizing probability (default: `0.1`)
- `--pairs`: comma-separated qubit pairs for 2-qubit Kraus noise (default: `0-1,1-2`)

## What `main.py` Does

1. Runs GHZ/W simulation once via `runner.run(...)`
2. Plots density matrix heatmaps with Plotly
3. Sweeps `p2` over a range via `runner.loop_run(...)`
4. Writes metrics to `out/loop_metrics.csv`
5. Generates multi-panel scatter comparisons from the CSV

## Programmatic Usage

```python
import numpy as np
from runner import run, loop_run
from datadump import dump_result_to_csv

results = run(bits=3, p1=0.01, p2=0.1, pairs=[(0, 1), (1, 2)])

sweep_values = np.arange(0.0, 0.62, 0.02, dtype=float)
loop_results = loop_run(
    bits=3,
    pairs=[(0, 1), (1, 2)],
    sweep_param="p2",
    sweep_values=sweep_values,
    p1=0.0,
)

df = dump_result_to_csv(loop_results, "out/loop_metrics.csv")
print(df.head())
```

## Output

- CSV sweep output (default): `out/loop_metrics.csv`
- Typical columns include `sweep_value` plus state-prefixed metrics, e.g.
  - `ghz_entropy_pure`, `w_entropy_pure`
  - `ghz_concurrence_ab_pure`, `w_concurrence_ab_pure`
  - `ghz_monogamy_pure`, `w_monogamy_pure`

## Notes

- `metrics.monogamy(...)` currently expects 3-qubit states.
- Plot windows depend on your local environment/Jupyter backend.

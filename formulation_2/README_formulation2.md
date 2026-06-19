# Formulation 2: Warm-guided annealing

This package implements Formulation 2, warm-guided annealing, for the Maximum Clique benchmark.
It follows the same repository layout as `formulation_1.zip`: place these files in a folder named `formulation_2` next to `base_setup/` and `ER_instances/`.

## Mathematical variant

The simulation uses the Hamiltonian

```text
H(t) = alpha(t) H_P + beta(t) H_D + eta(t) H_G
```

where

```text
H_D = - sum_i X_i
H_G = sum_i q_i Z_i
q_i = 2 c_i^* - 1
eta(t) = 4 rho s(1-s),   s = t/T
```

The warm-start vector `c^*` is obtained from the same SDP relaxation used in Formulation 1, then clipped to `[1e-6, 1 - 1e-6]` before constructing `q`.

The initial state remains `|+>^n`, because `eta(0)=0` and therefore `H(0)=H_D`. The final Hamiltonian remains the baseline problem Hamiltonian, because `eta(T)=0` and `H(T)=H_P`.

## Files

- `warm_guiding_annealing.py`: SDP warm-start construction, diagonal guiding-field construction, and continuous ODE simulation.
- `run_formulation2_experiment.py`: full experiment runner over the stored ER instances.
- `analysis_formulation2.py`: paired summary statistics, confidence intervals, and Wilcoxon tests.
- `make_plots_formulation2.py`: plots and regenerated interpretation tables.
- `README_formulation2.md`: this file.

## Run

From the repository root or from inside `formulation_2/`:

```bash
python formulation_2/run_formulation2_experiment.py
python formulation_2/make_plots_formulation2.py
```

To run a quick test:

```bash
python formulation_2/run_formulation2_experiment.py --limit 2 --n-steps 20 --rho-values 0 0.25 1.0 --skip-final-probabilities
python formulation_2/make_plots_formulation2.py
```

The default guiding strengths are

```text
rho in {0, 0.25, 0.50, 1.00, 2.00}
```

The value `rho=0` is included as a sanity check and should reproduce the baseline up to numerical tolerance.

## Outputs

The runner creates `formulation_2/results/` with:

```text
results/tables/formulation2_per_instance_results.csv
results/tables/formulation2_grouped_summary.csv
results/tables/formulation2_overall_summary.csv
results/tables/formulation2_bootstrap_ci.csv
results/raw/formulation2_energy_histories.csv
results/raw/formulation2_schedules.csv
results/raw/formulation2_warm_guiding_vectors.csv
results/raw/formulation2_final_probabilities.csv
results/logs/run_metadata.json
```

`formulation2_final_probabilities.csv` can become large. Disable it with:

```bash
python formulation_2/run_formulation2_experiment.py --skip-final-probabilities
```

The plotting script creates `formulation_2/results/plots/` with trajectory, schedule, paired-difference, scatter, heatmap, vector-distribution, and runtime plots.

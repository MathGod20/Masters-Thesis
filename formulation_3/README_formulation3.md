# Formulation 3: Block-mu driver annealing

This package implements the block-driver formulation with a maximum independent-block size `mu`.
The default tested values are:

```text
mu in {2, 3, 5}
```

For `mu = 1`, every block contains one vertex. The driver becomes the standard transverse-field driver and the closed-form initial state becomes `|+>^n`, so `mu = 1` recovers the baseline quantum annealing driver and initial state. The experiment script supports `mu = 1`, but it is not included in the default variant list because the baseline is already simulated separately.

## Idea

The vertices are partitioned into independent blocks. Each block has size at most `mu`. Since vertices inside the same block are pairwise non-adjacent, a clique can contain at most one vertex from each block.

For a block `B`, the local driver acts only on the block-valid states:

- no vertex selected from `B`
- exactly one vertex selected from `B`

The block driver connects the empty block state with each one-selected state. It never creates a state with two selected vertices inside the same block.

The ground state of each block driver is known in closed form. Therefore, the full initial state is the tensor product of the block ground states.

## Files

- `block_mu_annealing.py`: formulation implementation and ODE evolution.
- `run_formulation3_experiment.py`: runs the baseline once per instance and runs block-mu variants for each requested `mu`.
- `analysis_formulation3.py`: summary statistics and paired-difference analysis.
- `make_plots_formulation3.py`: plots and summary tables.
- `README_formulation3.md`: this file.

## Main defaults

```text
T = 10
N_steps = 100
alpha_k = k / N
beta_k = 1 - k / N
mu_values = 2 3 5
ODE method = DOP853
rtol = 1e-8
atol = 1e-10
```

The schedules for `alpha` and `beta`, the total annealing time `T`, and the number of intervals `N_steps` are kept equal to the baseline setup.

## Running the experiment

From the repository root, with `base_setup/` and `ER_instances/` available as siblings of `formulation_3/`:

```bash
python formulation_3/run_formulation3_experiment.py
```

Useful smaller test:

```bash
python formulation_3/run_formulation3_experiment.py --limit 2 --mu-values 2 3 5
```

To also check that `mu = 1` reproduces the baseline driver family:

```bash
python formulation_3/run_formulation3_experiment.py --limit 2 --mu-values 1 2 3 5
```

## Making plots

After running the experiment:

```bash
python formulation_3/make_plots_formulation3.py
```

## Outputs

The run script writes:

```text
formulation_3/results/tables/formulation3_per_instance_results.csv
formulation_3/results/tables/formulation3_grouped_summary.csv
formulation_3/results/tables/formulation3_overall_summary.csv
formulation_3/results/tables/formulation3_bootstrap_ci.csv
formulation_3/results/raw/formulation3_energy_histories.csv
formulation_3/results/raw/formulation3_schedules.csv
formulation_3/results/raw/formulation3_block_assignments.csv
formulation_3/results/logs/run_metadata.json
```

The plotting script writes figures into an overview folder and separate folders for every tested value of `mu`:

```text
formulation_3/results/plots/overview/
formulation_3/results/plots/by_mu/mu_2/
formulation_3/results/plots/by_mu/mu_3/
formulation_3/results/plots/by_mu/mu_5/
```

The overview folder compares all tested `mu` values against each other. The per-`mu` folders contain the important diagnostics for that specific variant, including trajectory plots, paired scatter plots, paired differences by graph size and density, runtime comparison, and block-structure diagnostics.

The plotting script also writes per-`mu` summary tables to:

```text
formulation_3/results/tables/by_mu/mu_2/
formulation_3/results/tables/by_mu/mu_3/
formulation_3/results/tables/by_mu/mu_5/
```

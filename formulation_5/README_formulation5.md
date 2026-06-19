# Formulation 5: Polynomial feedback continuous annealing

This folder contains the formulation-specific code for Formulation 5. The shared graph handling, QUBO construction, statevector simulation, and evaluation metrics are kept outside this folder in `base_setup/`.

Formulation 5 keeps the standard problem Hamiltonian and initial state, but replaces the fixed mixer schedule with a polynomial feedback rule for the mixer coefficient. The actual state evolution is simulated by continuous ODE integration, not by a Trotter product.

## Folder role

```text
formulation_5/
├── feedback_polynomial.py
├── analysis.py
├── run_formulation5_experiment.py
├── make_plots_formulation5.py
├── README.md
└── results/
```

The folder assumes the following repository-level structure:

```text
Formulations Repository/
├── ER_instances/
├── base_setup/
├── requirements.txt
└── formulation_5/
```

## Main idea

The continuous annealing Hamiltonian is

$$
H(t)=\alpha(t)H_P+\beta(t)H_M.
$$

The baseline uses the linear schedule

$$
\alpha_k=\frac{k}{N},
\qquad
\beta_k=1-\frac{k}{N},
\qquad
k=0,\ldots,N.
$$

Formulation 5 keeps the same $
\alpha_k$ values, but selects the next mixer value $
\beta_k$ using a local polynomial approximation of the expected problem energy. After $
\beta_k$ is selected, the state is evolved over the interval $[t_{k-1},t_k]$ by solving

$$
\frac{d\psi}{dt}
= -i\left(\alpha(t)H_P+\beta(t)H_M\right)\psi.
$$

Inside each interval, $
\alpha(t)$ and $
\beta(t)$ are linearly interpolated between their endpoint values.

## Files

`feedback_polynomial.py` implements the polynomial feedback rule and the feedback annealing simulation.

`analysis.py` builds summary tables, method rankings, confidence intervals, and best-method tables from the saved experiment results.

`run_formulation5_experiment.py` runs the baseline and all selected feedback variants on the ER benchmark instances.

`make_plots_formulation5.py` rebuilds interpretation tables and creates the overview and degree-specific plots.

## Install

Install dependencies from the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run the scripts from inside `formulation_5/`:

```bash
cd formulation_5
python run_formulation5_experiment.py
python make_plots_formulation5.py
```

A small smoke test can be run with:

```bash
python run_formulation5_experiment.py --limit 1 --method-limit 1 --n-steps 10 --total-time 1 --bootstrap-resamples 50 --ode-rtol 1e-6 --ode-atol 1e-8
python make_plots_formulation5.py
```

## Default thesis settings

```text
T = 10
N_steps = 100
polynomial degrees q = {2, 3, 4}
absolute restrictions rho = {1/N, 2/N, 5/N}
relative restrictions tau = {0.1, 0.05, 0.01, 0.005}
beta_0 = 1
beta_N = 0
epsilon_beta = 1e-8
ODE method = DOP853
ODE tolerances = rtol 1e-8, atol 1e-10
```

The default benchmark instances are read from:

```text
../ER_instances/
```

## Main metrics

The final state is evaluated directly from its statevector probabilities. The main metrics are:

```text
final_expected_energy
energy_variance
ground_energy
ground_state_probability
expected_feasible_energy
feasible_energy_variance
feasibility_probability
```

The energy variance is computed over the full final energy distribution:

$$
\operatorname{Var}(Q)=\sum_x p(x)\left(Q(x)-\mathbb{E}[Q]\right)^2.
$$

The feasible energy variance is computed after conditioning on feasible clique bitstrings:

$$
\operatorname{Var}(Q\mid x\in F)
=
\frac{\sum_{x\in F}p(x)\left(Q(x)-\mathbb{E}[Q\mid x\in F]\right)^2}
{\sum_{x\in F}p(x)}.
$$

## Outputs

The experiment writes tables to:

```text
results/tables/
```

and raw simulation outputs to:

```text
results/raw/
```

The most important output files are:

```text
results/tables/formulation5_per_instance_results.csv
results/tables/formulation5_grouped_summary.csv
results/tables/formulation5_method_ranking.csv
results/tables/formulation5_best_by_instance.csv
results/tables/formulation5_bootstrap_ci.csv
results/raw/formulation5_energy_histories.csv
results/raw/formulation5_beta_schedules.csv
results/logs/run_metadata.json
```

The plotting script creates:

```text
results/plots/overview/
results/plots/by_degree/q2/
results/plots/by_degree/q3/
results/plots/by_degree/q4/
```

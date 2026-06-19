# Formulation 1: Warm-start continuous annealing

This folder contains the formulation-specific code for Formulation 1. The shared graph handling, Maximum Clique QUBO construction, baseline statevector simulator, and evaluation metrics are kept outside this folder in `base_setup/`.

Formulation 1 changes the initial state and the mixer Hamiltonian. The Maximum Clique QUBO, the problem Hamiltonian, the total annealing time, the number of annealing intervals, the baseline endpoint schedule, the ODE solver, and the numerical tolerances remain aligned with the shared setup.

## Folder role

```text
formulation_1/
├── warm_start_annealing.py
├── analysis_formulation1.py
├── run_formulation1_experiment.py
├── make_plots_formulation1.py
├── README_formulation1.md
└── results/
```

The folder assumes the following repository-level structure:

```text
Formulations Repository/
├── ER_instances/
├── base_setup/
├── requirements.txt
├── formulation_1/
└── formulation_2/
```

## Main idea

The baseline annealing process starts from the uniform superposition and uses the standard transverse-field mixer

$$
H_M=-\sum_{i=1}^n X_i.
$$

Formulation 1 solves an SDP relaxation of the baseline Maximum Clique QUBO and obtains a fractional vector

$$
c=(c_1,\ldots,c_n), \qquad c_i\in[0,1].
$$

After clipping numerical boundary values, the vector defines the warm-start product state

$$
|\psi_0^{WS}\rangle
=
\bigotimes_{i=1}^n
\left(\sqrt{1-c_i}|0\rangle+\sqrt{c_i}|1\rangle\right).
$$

Equivalently, for each qubit,

$$
\theta_i=2\arcsin(\sqrt{c_i}).
$$

The corresponding warm-start mixer is

$$
H_M^{WS}=\sum_{i=1}^n H_{M,i}^{WS},
$$

with

$$
H_{M,i}^{WS}
=
-\sin(\theta_i)X_i-\cos(\theta_i)Z_i.
$$

The warm-start annealing Hamiltonian is therefore

$$
H^{WS}(t)=\alpha(t)H_P+\beta(t)H_M^{WS}.
$$

The endpoint schedule is kept equal to the baseline schedule:

$$
\alpha_k=\frac{k}{N},
\qquad
\beta_k=1-\frac{k}{N},
\qquad
k=0,\ldots,N.
$$

Inside each interval, $\alpha(t)$ and $\beta(t)$ are linearly interpolated between their endpoint values. The state is propagated by solving

$$
\frac{d\psi}{dt}
=
-i\left(\alpha(t)H_P+\beta(t)H_M^{WS}\right)\psi
$$

with `solve_ivp` using the `DOP853` method.

## Files

`warm_start_annealing.py` implements the SDP warm-start vector, the warm-start state, the warm-start mixer, and the ODE-based warm-start annealing simulation.

`analysis_formulation1.py` builds summary tables, confidence intervals, and overall comparison tables from the saved experiment results.

`run_formulation1_experiment.py` runs the baseline and Formulation 1 on the ER benchmark instances.

`make_plots_formulation1.py` rebuilds interpretation tables and creates the Formulation 1 plots.

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

Run the scripts from the repository root:

```bash
python formulation_1/run_formulation1_experiment.py
python formulation_1/make_plots_formulation1.py
```

A small smoke test can be run with:

```bash
python formulation_1/run_formulation1_experiment.py --limit 1 --n-steps 10 --total-time 1 --bootstrap-resamples 50 --ode-rtol 1e-6 --ode-atol 1e-8
python formulation_1/make_plots_formulation1.py
```

## Default thesis settings

```text
T = 10
N_steps = 100
alpha_k = k / N_steps
beta_k = 1 - k / N_steps
ODE method = DOP853
ODE tolerances = rtol 1e-8, atol 1e-10
SDP solver = SCS
SDP tolerance = 1e-6
SCS maximum iterations = 20000
warm-start clipping epsilon = 1e-6
```

The default benchmark instances are read from:

```text
../ER_instances/
```

relative to the `formulation_1/` folder.

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
\operatorname{Var}(Q)
=
\sum_x p(x)\left(Q(x)-\mathbb{E}[Q]\right)^2.
$$

The feasible energy variance is computed after conditioning on feasible clique bitstrings:

$$
\operatorname{Var}(Q\mid x\in F)
=
\frac{
\sum_{x\in F}p(x)\left(Q(x)-\mathbb{E}[Q\mid x\in F]\right)^2
}{
\sum_{x\in F}p(x)
}.
$$

The paired deltas are computed as warm-start minus baseline:

```text
delta_energy
delta_energy_variance
delta_p_ground
delta_expected_feasible_energy
delta_feasible_energy_variance
delta_p_feas
```

For energy-based metrics, smaller values are better. For probability-based metrics, larger values are better.

## Outputs

The experiment writes tables to:

```text
formulation_1/results/tables/
```

The most important table outputs are:

```text
formulation_1/results/tables/formulation1_per_instance_results.csv
formulation_1/results/tables/formulation1_grouped_summary.csv
formulation_1/results/tables/formulation1_overall_summary.csv
formulation_1/results/tables/formulation1_bootstrap_ci.csv
```

Raw simulation outputs are written to:

```text
formulation_1/results/raw/
```

The most important raw outputs are:

```text
formulation_1/results/raw/formulation1_energy_histories.csv
formulation_1/results/raw/formulation1_schedules.csv
formulation_1/results/raw/formulation1_warm_start_vectors.csv
```

Run metadata are written to:

```text
formulation_1/results/logs/run_metadata.json
```

The plotting script writes figures to:

```text
formulation_1/results/plots/
```

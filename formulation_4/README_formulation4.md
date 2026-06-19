# Formulation 4: QAOA-mixer-assisted annealing

This folder contains the code for Formulation 4.

The formulation keeps the baseline initial state, the baseline penalized Maximum Clique problem Hamiltonian, and the standard transverse-field driver. The structural change is the addition of a graph-aware QAOA-style mixer as a third Hamiltonian term during the anneal.

The simulated Hamiltonian is

$$
H(t)
=
\alpha(s)H_P
+
\beta(s)H_D
+
\delta(s)H_M^{\mathrm{QAOA}},
\qquad
s=\frac{t}{T}.
$$

The problem schedule is linear:

$$
\alpha(s)=s.
$$

The standard-driver schedule follows the baseline schedule over the whole annealing time:

$$
\beta(s)=1-s.
$$

The auxiliary coefficient is

$$
\delta(s)=4\rho s(1-s),
$$

where \(\rho\) controls the maximum strength of the auxiliary mixer. The auxiliary term is inactive at the beginning and at the end of the anneal, and it reaches its maximum value \(\rho\) at \(s=1/2\).

The tested values are

$$
\rho\in\{0.25,0.50,1.00,2.00\}.
$$

The code tests these values directly. There is no additional relative-strength parameter and no instance-dependent rescaling of \(\rho\). The spectral norm of the auxiliary mixer is still recorded as diagnostic information, but it is not used to convert the tested strength values.

## Folder structure

```text
formulation_4/
├── analysis_formulation4.py
├── qaoa_mixer_assisted_annealing.py
├── make_plots_formulation4.py
├── README_formulation4.md
└── run_formulation4_experiment.py
```

## Files

`qaoa_mixer_assisted_annealing.py` contains the formulation-specific Hamiltonian-vector products, the baseline problem and driver schedules, the auxiliary schedule, and the ODE evolution.

`run_formulation4_experiment.py` runs the baseline and Formulation 4 on the repository-level `ER_instances/` folder.

`analysis_formulation4.py` contains paired comparison summaries, confidence intervals, improvement rates, and best-strength diagnostics.

`make_plots_formulation4.py` creates PNG figures from the saved result tables.

## Inputs

The code expects the repository to have this structure:

```text
Formulations Repository/
├── base_setup/
├── ER_instances/
├── formulation_4/
├── generate_instances.py
└── requirements.txt
```

The shared files in `base_setup/` are used for reading graph instances, constructing the baseline QUBO, computing statevector metrics, and running the baseline continuous ODE annealing simulation.

## Run the experiment

From the repository root, run:

```bash
python formulation_4/run_formulation4_experiment.py
```

This saves the main results in:

```text
formulation_4/results/
```

To change the tested auxiliary strengths, use:

```bash
python formulation_4/run_formulation4_experiment.py --rho-values 0.25 0.5 1.0 2.0
```

## Run the plots

After the experiment finishes, run:

```bash
python formulation_4/make_plots_formulation4.py
```

This saves PNG figures in:

```text
formulation_4/figures/
```

## Main outputs

```text
formulation_4/results/formulation4_results.csv
formulation_4/results/energy_histories.csv
formulation_4/results/schedules.csv
formulation_4/results/summary_by_rho.csv
formulation_4/results/summary_by_rho_and_n.csv
formulation_4/results/summary_by_rho_and_p.csv
formulation_4/results/confidence_intervals.csv
formulation_4/results/best_by_instance.csv
formulation_4/results/auxiliary_mixer_specification.csv
formulation_4/results/run_metadata.json
```

The result table saves the tested auxiliary mixer strength \(\rho\) directly.

from base_setup.instances import generate_instance_grid


if __name__ == "__main__":
    paths = generate_instance_grid(
        output_dir="ER_instances",
        n_values=[6, 9, 12],
        p_values=[0.1, 0.3, 0.5, 0.7],
        replicates=10,
        base_seed=2026,
    )
    print(f"Generated {len(paths)} instances in ER_instances.")

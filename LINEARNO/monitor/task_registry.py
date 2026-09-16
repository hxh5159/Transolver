"""Dataset names understood by the monitor wrapper."""

DATASET_ALIASES = {
    "airfoil": "standard_airfoil",
    "standard_airfoil": "standard_airfoil",
    "pipe": "pipe",
    "plasticity": "plasticity",
    "plas": "plasticity",
    "navier_stokes": "navier_stokes",
    "ns": "navier_stokes",
    "darcy": "darcy",
    "elasticity": "elasticity",
    "elas": "elasticity",
    "airfrans": "airfrans",
    "shapenetcar": "shapenetcar",
    "car": "shapenetcar",
}

ENTRYPOINT_HINTS = {
    "standard_airfoil": "PDE-Solving-StandardBenchmark/exp_airfoil.py",
    "pipe": "PDE-Solving-StandardBenchmark/exp_pipe.py",
    "plasticity": "PDE-Solving-StandardBenchmark/exp_plas.py",
    "navier_stokes": "PDE-Solving-StandardBenchmark/exp_ns.py",
    "darcy": "PDE-Solving-StandardBenchmark/exp_darcy.py",
    "elasticity": "PDE-Solving-StandardBenchmark/exp_elas.py",
    "airfrans": "Airfoil-Design-AirfRANS/main.py",
    "shapenetcar": "Car-Design-ShapeNetCar/main.py",
}


def canonical_dataset(value):
    key = value.strip().lower().replace("-", "_")
    try:
        return DATASET_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(set(DATASET_ALIASES.values())))
        raise ValueError(f"unknown dataset {value!r}; expected one of: {choices}") from exc

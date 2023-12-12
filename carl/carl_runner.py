import os
import argparse
import importlib.util
import importlib
import sys
sys.setrecursionlimit(8000)
import tempfile

from carl.backends.local_sequential import LocalSequentialBackend
from carl.experiments.experiment import Experiment
import joblib as jl


def load_experiment(experiment_path: str) -> Experiment:
    if experiment_path.endswith('.joblib'):
        return jl.load(experiment_path)

    temp_name = next(tempfile._get_candidate_names())
    spec = importlib.util.spec_from_file_location(temp_name, experiment_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[temp_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, 'experiment'):
        raise ValueError(f"Experiment file {experiment_path} does not contain an experiment variable")

    return module.experiment


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", type=str, choices=["local-sequential"], default="local-sequential")
    parser.add_argument(
        "--experiment", type=str, help='Path to experiment config file or path to experiment.joblib file')
    args = parser.parse_args()

    backend = {
        'local-sequential': LocalSequentialBackend,
    }[args.backend]()

    experiment = load_experiment(args.experiment)

    # Set experiment attributes
    setattr(experiment, '_backend', args.backend)
    setattr(experiment, '_experiment_path', args.experiment)

    # Check if experiment is valid with respect to the backend
    backend.validate_experiment(experiment)

    # Deploy the experiment
    backend.run_experiment(experiment)

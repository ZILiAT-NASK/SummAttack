from carl.experiments.experiment import Experiment
from carl.backends.backend import Backend
import os
from carl.gin.config import make_gin_bindings
import joblib as jl
import gin
import sys
from jobs import FullPipeline
sys.setrecursionlimit(8000)


class LocalSequentialBackend(Backend):

    def __init__(self) -> None:
        super().__init__()

    def validate_experiment(self, experiment: Experiment) -> None:

        experiment_to_run = experiment
        experiment.base_config = experiment_to_run.base_config

    def run_experiment(self, experiment: Experiment) -> None:

        # Parse gin config
        gin_bindings = make_gin_bindings(experiment.base_config)
        gin.parse_config_files_and_bindings(['configs/empty.gin'], gin_bindings)

        print(gin_bindings)

        # Log experiment hyperparameters
        storage_path = 'storage'
        os.makedirs(storage_path, exist_ok=True)
        tmp_config = experiment.base_config.copy()
        tmp_config['pwd'] = os.getcwd()
        tmp_config = {f'hyperparams/{k}': v for k, v in tmp_config.items()}
        jl.dump(tmp_config, os.path.join(storage_path, 'hyperparams.attr'))

        # Run experiment
        job = FullPipeline()
        job.run()




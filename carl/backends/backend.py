from abc import ABC, abstractmethod
from carl.experiments.experiment import Experiment


class Backend(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def run_experiment(self, experiment: Experiment) -> None:
        pass

    @abstractmethod
    def validate_experiment(self, experiment: Experiment) -> None:
        pass

# SummAttack

![Thumbnail](https://github.com/ZILiAT-NASK/SummAttack/blob/main/images/SummAttack.png)

SummAttack is an open-source framework designed for conducting adversarial attacks on large language models specifically tailored for the summarization task.


## Running the pipeline

To run the pipeline, you need to fill the config file (examples in the `configs` directory) with chosen attacks, metrics and parameters. Then, you can run the pipeline with the following command:

```
python3 -m carl.carl_runner --experiment <path_to_config_file>
```

## Attacks

To add your own attack you need to add inherit from the `Attack` class and implement the `attack` method. Then, you need to add your attack in the `attacks/__init__.py` file.

Each attack should have a unique name. In the attack function you should return the adversarial example and the number of changes made to the input. The adversarial example should be a string with the same number of sentences as the input. The number of changes should be an integer.

```
class Attack(ABC):
    def __init__(self):
        self.name = None

    def attack(self, sentences: List[str]) -> Tuple[str, int]:
        changes = 0
        return ' '.join(sentences), changes

    def __call__(self, sentences: List[str]) -> str:
        return self.attack(sentences)

    def __name__(self):
        return self.name
        
```

## Metrics

To add your own metric you need to add inherit from the `Metric` class and implement the `compute` method. Then, you need to add your metric in the `metrics/__init__.py` file.

Each metric should have a unique name. In the compute function you should return the value of the metric. The value should be a float.

There are two types of metrics: `MetricSummarytoSummary` and `MetricOriginalTextToSummary`. The first type of metric takes two summaries as input and the second type takes a summary and the original text as input. 

```
class MetricSummarytoSummary(ABC):
    def __init__(self):
        self.name = None
        self.type = None

    def compute(self, generated_summary: str, reference_summary: str) -> float:
        pass

    def __call__(self, generated_summary: str, reference_summary: str) -> float:
        return self.compute(generated_summary, reference_summary)

    def __name__(self):
        return self.name


class MetricOriginalTextToSummary(ABC):
    def __init__(self):
        self.name = None
        self.type = None

    def compute(self, generated_summary: str, original_text: str) -> float:
        pass

    def __call__(self, generated_summary: str, original_text: str) -> float:
        return self.compute(generated_summary, original_text)

    def __name__(self):
        return self.name

```

## Summarizers

The most popular commercial APIs (OpenAI) and open-source libraries (HuggingFace) are supported. To add your own summarizer you need to add inherit from the `Summarizer` class and implement the `summarize` method. Then, you need to add your summarizer in the `summarizers/__init__.py` file.

```
class Summarizer(ABC):
    def __init__(self):
        self.name = None

    def generate_summary(self, text: str) -> str:
        pass

    def __call__(self, text: str) -> str:
        return self.summarize(text)

    def __name__(self):
        return self.name
        
```
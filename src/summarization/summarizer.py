from abc import ABC


class Summarizer(ABC):
    def __init__(self):
        self.name = None

    def generate_summary(self, text: str) -> str:
        pass

    def __call__(self, text: str) -> str:
        return self.summarize(text)

    def __name__(self):
        return self.name
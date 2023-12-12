from abc import ABC
from factsumm import FactSumm
import stylo_metrix
import spacy
from typing import List, Dict
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize


class MetricOriginalTextToSummary(ABC):
    def __init__(self):
        self.name = None
        self.type = None

    def compute(self, generated_summary: str, original_text: str) -> float:
        pass

    def __call__(self, generated_summary: str, original_text: str) -> float:
        return self.compute(generated_summary, original_text)


class FactSummScore(MetricOriginalTextToSummary):
    def __init__(self):
        super().__init__()
        self.name = "FactSummScore"
        self.type = "factuality"
        self.metric = FactSumm()

    def compute(self, generated_summary: str, original_text: str) -> float:
        try:
            return self.metric(original_text, generated_summary)
        except Exception as e:
            print(f"Error in FactSummScore, {e}")
            return 0.0


class NovelNGrams(MetricOriginalTextToSummary):
    def __init__(self, n=None):
        super().__init__()
        if n is None:
            n = [3, 4, 5]
        self.name = "NovelNGrams"
        self.type = "occurrence"
        self.n = n

    def compute(self, generated_summary: str, original_text: str) -> dict[int, int]:
        novel_ngrams_count = dict()
        original_tokens = word_tokenize(original_text)
        summary_tokens = word_tokenize(generated_summary)
        for n in self.n:
            # Generate n-grams for the original text and the generated summary
            original_ngrams = list(ngrams(original_tokens, n))
            summary_ngrams = list(ngrams(summary_tokens, n))

            # Count the occurrences of n-grams in the original text and the generated summary
            original_ngram_counts = Counter(original_ngrams)
            summary_ngram_counts = Counter(summary_ngrams)

            # Compute the number of novel n-grams
            novel_ngrams = 0
            for ngram, count in summary_ngram_counts.items():
                if original_ngram_counts[ngram] < count:
                    novel_ngrams += 1

            novel_ngrams_count[n] = novel_ngrams

        return novel_ngrams_count


class Stylometrix(MetricOriginalTextToSummary):
    name = "Stylometrix"

    def __init__(self):
        super().__init__()
        self.name = "Stylometrix"
        self.type = "stylometry"
        nlp = spacy.load("en_core_web_trf")
        self.sm = stylo_metrix.StyloMetrix('eng')
        self.nlp = self.sm._nlp

    def compute(self, text:str) -> str:
        stylo = self.sm.transform(text)
        return stylo


class Cohmetrix(MetricOriginalTextToSummary):
    name = "Cohmetrix"

    def __init__(self):
        super().__init__()
        self.name = "Cohmetrix"
        self.type = "coherence"

    def compute(self, generated_summary: str, original_text: str) -> float:
        # TODO: implement
        pass

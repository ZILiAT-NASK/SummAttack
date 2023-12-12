from abc import ABC
from rouge import Rouge
import bert_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import pipeline


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


class Rouge1(MetricSummarytoSummary):
    def __init__(self):
        super().__init__()
        self.name = "Rouge1"
        self.type = "occurrence"
        self.rouge = Rouge()

    def compute(self, generated_summary: str, reference_summary: str) -> float:
        scores = self.rouge.get_scores(generated_summary, reference_summary)
        return scores[0]["rouge-1"]["f"]


class Rouge2(MetricSummarytoSummary):
    def __init__(self):
        super().__init__()
        self.name = "Rouge2"
        self.type = "occurrence"
        self.rouge = Rouge()

    def compute(self, generated_summary: str, reference_summary: str) -> float:
        scores = self.rouge.get_scores(generated_summary, reference_summary)
        return scores[0]["rouge-2"]["f"]


class RougeL(MetricSummarytoSummary):
    def __init__(self):
        super().__init__()
        self.name = "RougeL"
        self.type = "occurrence"
        self.rouge = Rouge()

    def compute(self, generated_summary: str, reference_summary: str) -> float:
        scores = self.rouge.get_scores(generated_summary, reference_summary)
        return scores[0]["rouge-l"]["f"]


class BERTScore(MetricSummarytoSummary):
    def __init__(self):
        super().__init__()
        self.name = "BERTScore"
        self.type = "generation"

    def compute(self, generated_summary: str, reference_summary: str) -> float:
        return bert_score.score([generated_summary], [reference_summary], lang='en')[2].item()


class SentimentScore(MetricSummarytoSummary):
    def __init__(self):
        super().__init__()
        self.name = "SentimentScore"
        self.type = "sentiment"
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.sentiment_classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def predict_class(self, text: str):
        return self.sentiment_classifier(text)[0]

    def compute(self, generated_summary: str, reference_summary: str) -> float:
        scores_referece = self.predict_class(reference_summary)
        scores_generated = self.predict_class(generated_summary)
        if scores_referece["label"] == scores_generated["label"]:
            return 1
        else:
            return 0


class NamedEntitiesScore(MetricSummarytoSummary):
    def __init__(self):
        super().__init__()
        self.name = "NamedEntitiesScore"
        self.type = "named_entities"
        ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)

    def number_of_named_entities(self, summary: str) -> int:
        number_of_ne = 0
        entities = self.ner(summary)
        for entity in entities:
            if entity["entity"] != "O":
                number_of_ne += 1
        return number_of_ne

    def compute(self, generated_summary: str, reference_summary: str) -> float:
        number_of_ne_reference = self.number_of_named_entities(reference_summary)
        number_of_ne_generated = self.number_of_named_entities(generated_summary)
        return number_of_ne_generated / number_of_ne_reference

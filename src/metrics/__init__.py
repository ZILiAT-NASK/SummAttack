import gin
from src.metrics import metrics_summary_to_summary, metrics_summary_to_text


def configure_class(class_object):
    return gin.external_configurable(class_object, module='metrics')


Rouge1 = configure_class(metrics_summary_to_summary.Rouge1)
Rouge2 = configure_class(metrics_summary_to_summary.Rouge2)
RougeL = configure_class(metrics_summary_to_summary.RougeL)
BERTScore = configure_class(metrics_summary_to_summary.BERTScore)
FactSummScore = configure_class(metrics_summary_to_text.FactSummScore)
SentimentScore = configure_class(metrics_summary_to_summary.SentimentScore)
NamedEntitiesScore = configure_class(metrics_summary_to_summary.NamedEntitiesScore)
NovelNGrams = configure_class(metrics_summary_to_text.NovelNGrams)
Stylometrix = configure_class(metrics_summary_to_text.Stylometrix)
Cohmetrix = configure_class(metrics_summary_to_text.Cohmetrix)
MetricSummarytoSummary = configure_class(metrics_summary_to_summary.MetricSummarytoSummary)
MetricOriginalTextToSummary = configure_class(metrics_summary_to_text.MetricOriginalTextToSummary)
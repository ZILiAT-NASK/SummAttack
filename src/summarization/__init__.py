import gin
from src.summarization.summarizer import Summarizer
from src.summarization.gpt_summarizer import GPTSummarizer
from src.summarization.chat_gpt_summarizer import ChatGPTSummarizer
from src.summarization.hf_summarizer import HfSummarizer


def configure_class(class_object):
    return gin.external_configurable(class_object, module='summarizers')


Summarizer = configure_class(Summarizer)
GPTSummarizer = configure_class(GPTSummarizer)
ChatGPTSummarizer = configure_class(ChatGPTSummarizer)
HfSummarizer = configure_class(HfSummarizer)

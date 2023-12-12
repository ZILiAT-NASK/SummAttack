from src.summarization import Summarizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


class HfSummarizer(Summarizer):
    def __init__(self, prompt: str, temperature: float, model_name: str = "t5-base"):
        super().__init__()
        self.name = "Huggingface Summarizer"
        self.prompt = prompt
        self.temperature = temperature
        self.model = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(self.model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            )        

    def generate_summary(self, text: str) -> str:
        text_length = len(text)
        prompt_length = len(self.prompt)
        if text_length + prompt_length > 4097:
            text = text[:4097 - prompt_length]

        full_prompt = f"{self.prompt} {text} \n"

        sequences = self.pipeline(
            full_prompt,
            max_length=200,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            )

        summary = sequences[0]["generated_text"].strip()
        return summary

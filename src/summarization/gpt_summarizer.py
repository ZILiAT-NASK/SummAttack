from src.summarization import Summarizer
import openai


class GPTSummarizer(Summarizer):
    def __init__(self, prompt: str, temperature: float, model_name: str, openai_apikey: str):
        super().__init__()
        self.name = "GPT Summarizer"
        self.prompt = prompt
        self.temperature = temperature
        self.model_name = model_name
        self.openai_apikey = openai_apikey

    def generate_summary(self, text: str) -> str:
        text_length = len(text)
        prompt_length = len(self.prompt)
        if text_length + prompt_length > 4097:
            text = text[:4097 - prompt_length]

        full_prompt = f"{self.prompt} {text} \n"

        openai.api_key = self.openai_apikey

        response = openai.Completion.create(
            engine=self.model_name,
            prompt=full_prompt,
            temperature=self.temperature,
            max_tokens=80,
            top_p=1,
            n=1,
            stop=None,
            frequency_penalty=0,
            presence_penalty=0,
        )
        summary = response.choices[0].text.strip()
        return summary

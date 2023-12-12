from src.summarization import Summarizer
import openai


class ChatGPTSummarizer(Summarizer):
    def __init__(self, prompt: str, temperature: float, openai_apikey: str):
        super().__init__()
        self.name = "Chat GPT Summarizer"
        self.prompt = prompt
        self.temperature = temperature
        self.openai_apikey = openai_apikey

    def generate_summary(self, text: str) -> str:
        text_length = len(text)
        prompt_length = len(self.prompt)
        if text_length + prompt_length > 4097:
            text = text[:4097 - prompt_length]

        full_prompt = f"{self.prompt} {text} \n"

        openai.api_key = self.openai_apikey

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{full_prompt}"},
            ],
            temperature=self.temperature,
            max_tokens=50,
            n=1,
        )
        summary = response.choices[0]['message']['content']
        return summary

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class HFLLM:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name
        self._load()

    def _load(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU only
        )

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        output = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return output[0]["generated_text"].strip()


if __name__ == "__main__":
    llm = HFLLM()
    test_prompt = "Answer in one sentence: Do we pay for routine eye examinations?"
    print(llm.generate(test_prompt))

import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from openai import OpenAI

class Pipeline:
    class Valves(BaseModel):
        MODEL_NAME: str = Field(default="gpt-4", description="LLM model to use")
        OPENAI_API_KEY: str = Field(default=os.getenv("OPENAI_API_KEY", "sk-abc"), description="API key for OpenAI")
        VLLM_HOST: str = Field(default=os.getenv("VLLM_HOST", "http://localhost:8000/v1"), description="LLM host endpoint")

    def __init__(self):
        self.name = "AI Teaching Step 1"
        self.valves = self.Valves()

    async def on_startup(self):
        print(f"[STARTUP] {self.name} pipeline started.")

    async def on_shutdown(self):
        print(f"[SHUTDOWN] {self.name} pipeline stopped.")

    def get_llm(self):
        return OpenAI(
            base_url=self.valves.VLLM_HOST,
            api_key=self.valves.OPENAI_API_KEY
        )

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:

        topic = user_message.strip()
        print(f"[MESSAGE] Topic: {topic}")

        llm = self.get_llm()
        prompt = f"""
You are a teaching assistant AI. The user wants to learn about: \"{topic}\".
Generate a clear, step-by-step learning outline (3 to 5 steps) with bullet points. Just return a plain text list.
"""

        response = llm.chat.completions.create(
            model=self.valves.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        outline = response.choices[0].message.content.strip()
        print(f"[OUTLINE] {outline}")

        return f"📘 Here’s how we’ll explore **{topic}**:\n\n{outline}"

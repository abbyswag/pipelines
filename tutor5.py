import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from openai import OpenAI

class Pipeline:
    class Valves(BaseModel):
        MODEL_NAME: str = Field(default="gpt-4", description="LLM model to use")
        OPENAI_API_KEY: str = Field(default=os.getenv("OPENAI_API_KEY", "sk-abc"), description="API key for OpenAI")
        VLLM_HOST: str = Field(default=os.getenv("VLLM_HOST", "http://localhost:8000/v1"), description="LLM host endpoint")
        MAX_STEPS: int = Field(default=5, description="Max steps to explain")

    def __init__(self):
        self.name = "AI Teaching Step 2"
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
        llm = self.get_llm()
        print(f"[TOPIC] {topic}")

        # Step 1: Generate outline
        outline_prompt = f"""
You are a teaching assistant AI. The user wants to learn about: "{topic}".
Generate a step-by-step learning outline (max {self.valves.MAX_STEPS} steps) with short titles only.
Return a plain text list, one item per line.
"""
        outline_response = llm.chat.completions.create(
            model=self.valves.MODEL_NAME,
            messages=[{"role": "user", "content": outline_prompt}]
        )

        outline_text = outline_response.choices[0].message.content.strip()
        print(f"[OUTLINE]\n{outline_text}")

        steps = [line.strip("0123456789. ").strip() for line in outline_text.splitlines() if line.strip()]
        if not steps:
            return "❌ Failed to generate an outline. Please try again."

        yield f"📘 Here's how we'll explore **{topic}**:\n\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])

        # Step 2: Explain each step
        for i, step in enumerate(steps):
            explain_prompt = f"""
You are an expert tutor. Explain the following learning step in simple, clear, and engaging language for beginners:\n\nStep: \"{step}\"
"""
            explain_response = llm.chat.completions.create(
                model=self.valves.MODEL_NAME,
                messages=[{"role": "user", "content": explain_prompt}]
            )
            explanation = explain_response.choices[0].message.content.strip()
            print(f"[STEP {i+1}] {step}\n{explanation}\n")
            yield f"\n---\n\n### Step {i+1}: {step}\n{explanation}"

        yield "\n✅ End of learning session. Ask me if you want visuals or deeper info!"

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
        self.name = "Tutor"
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
You are a brilliant domain expert and educator AI. The user wants to deeply understand: \"{topic}\".
Generate a teaching outline with up to {self.valves.MAX_STEPS} conversational, engaging steps.
Use beginner-friendly titles. One step per line.
"""
        outline_response = llm.chat.completions.create(
            model=self.valves.MODEL_NAME,
            messages=[{"role": "user", "content": outline_prompt}]
        )
        steps_raw = outline_response.choices[0].message.content.strip()
        steps = [line.strip("0123456789. ").strip() for line in steps_raw.splitlines() if line.strip()]

        yield f"📘 Here's how we'll explore **{topic}** step-by-step:
\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])

        # Step 2: Explain each step conversationally
        for i, step in enumerate(steps):
            explain_prompt = f"""
You are an expert tutor in the topic: \"{topic}\". You're speaking directly to a student who wants to understand the concept of:
\n"{step}"

Explain this step in a friendly, conversational way.
Use simple language, analogies, and examples to help the user deeply understand it.
Ask occasional questions back to the user to engage them.
"""
            explain_response = llm.chat.completions.create(
                model=self.valves.MODEL_NAME,
                messages=[{"role": "user", "content": explain_prompt}]
            )
            explanation = explain_response.choices[0].message.content.strip()

            yield f"""
---

### Step {i+1}: {step}
{explanation}
"""

        yield "\n✅ That's it for now! Let me know if you'd like to dive deeper into any step or topic."
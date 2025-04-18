import os
import uuid
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from openai import OpenAI

class Pipeline:
    class Valves(BaseModel):
        MODEL_NAME: str = Field(default="gpt-4", description="LLM model to use")
        OPENAI_API_KEY: str = Field(default=os.getenv("OPENAI_API_KEY", "sk-abc"), description="API key for OpenAI")
        VLLM_HOST: str = Field(default=os.getenv("VLLM_HOST", "http://localhost:8000/v1"), description="LLM host endpoint")
        MAX_STEPS: int = Field(default=5, description="Max steps to explain")
        OUTPUT_DIR: str = Field(default="/app/pipelines/outputs", description="Folder to save HTML")
        PUBLIC_HOST: str = Field(default="localhost", description="Public URL host")
        PUBLIC_PORT: int = Field(default=9099, description="Public URL port")

    def __init__(self):
        self.name = "AI Teaching Step 3"
        self.valves = self.Valves()
        os.makedirs(self.valves.OUTPUT_DIR, exist_ok=True)

    async def on_startup(self):
        print(f"[STARTUP] {self.name} pipeline started.")

    async def on_shutdown(self):
        print(f"[SHUTDOWN] {self.name} pipeline stopped.")

    def get_llm(self):
        return OpenAI(
            base_url=self.valves.VLLM_HOST,
            api_key=self.valves.OPENAI_API_KEY
        )

    def generate_html_animation(self, step_title: str, visual_prompt: str) -> str:
        prompt = f"""
Create a beautiful, animated HTML page to help explain:
\"{visual_prompt}\"
Use advanced web animations using Three.js, Anime.js, or Lottie.
Return full HTML (with embedded JS/CSS), self-contained.
"""
        llm = self.get_llm()
        response = llm.chat.completions.create(
            model=self.valves.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        html_code = response.choices[0].message.content.strip()
        if "<html" not in html_code:
            html_code = f"""
<!DOCTYPE html>
<html>
<head><title>{step_title}</title></head>
<body>
{html_code}
</body>
</html>
"""

        filename = f"step_{step_title.replace(' ', '_').lower()}_{uuid.uuid4().hex[:6]}.html"
        filepath = os.path.join(self.valves.OUTPUT_DIR, filename)
        with open(filepath, "w") as f:
            f.write(html_code)

        return f"http://{self.valves.PUBLIC_HOST}:{self.valves.PUBLIC_PORT}/outputs/{filename}"

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
Generate a learning outline with max {self.valves.MAX_STEPS} steps (just titles).
Return each step as a new line.
"""
        outline_response = llm.chat.completions.create(
            model=self.valves.MODEL_NAME,
            messages=[{"role": "user", "content": outline_prompt}]
        )
        steps_raw = outline_response.choices[0].message.content.strip()
        steps = [line.strip("0123456789. ").strip() for line in steps_raw.splitlines() if line.strip()]

        yield f"📘 Here's how we'll explore **{topic}**:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])

        # Step 2 & 3: Explain each step + HTML
        for i, step in enumerate(steps):
            explain_prompt = f"""
You are an expert tutor. Explain the following learning step clearly and simply:\n\nStep: \"{step}\"
"""
            explain_response = llm.chat.completions.create(
                model=self.valves.MODEL_NAME,
                messages=[{"role": "user", "content": explain_prompt}]
            )
            explanation = explain_response.choices[0].message.content.strip()

            # Step 3: Generate animation
            html_link = self.generate_html_animation(step_title=step, visual_prompt=step)

            yield f"""
---

### Step {i+1}: {step}
{explanation}

👉 [View Animation]({html_link})
"""

        yield "\n✅ That’s the end! Let me know if you want deeper dives or more visualizations!"

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

    def __init__(self):
        self.name = "Tutor"
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
Create a fully self-contained HTML page that visualizes the concept:
\"{visual_prompt}\"
Use animations (Three.js, Anime.js, or Lottie).
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

        return html_code  # Return full HTML to embed in chat

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
You are a teaching assistant AI. The user wants to learn about: \"{topic}\".
Generate a learning outline with max {self.valves.MAX_STEPS} steps (just titles).
Return each step on a new line.
"""
        outline_response = llm.chat.completions.create(
            model=self.valves.MODEL_NAME,
            messages=[{"role": "user", "content": outline_prompt}]
        )
        steps_raw = outline_response.choices[0].message.content.strip()
        steps = [line.strip("0123456789. ").strip() for line in steps_raw.splitlines() if line.strip()]

        yield f"📘 Here's how we'll explore **{topic}**:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])

        # Step 2 & 3: Explain each step + return HTML inline
        for i, step in enumerate(steps):
            explain_prompt = f"""
You are an expert tutor. Explain the following learning step clearly and simply:\n\nStep: \"{step}\"
"""
            explain_response = llm.chat.completions.create(
                model=self.valves.MODEL_NAME,
                messages=[{"role": "user", "content": explain_prompt}]
            )
            explanation = explain_response.choices[0].message.content.strip()

            # Generate HTML animation (raw content)
            html_code = self.generate_html_animation(step_title=step, visual_prompt=step)

            yield f"""
---

### Step {i+1}: {step}
{explanation}

#### 🖼️ Visualization (HTML)
```html
{html_code}
```
"""

        yield "\n✅ Done! You can now render these HTMLs or convert them into visual elements in your frontend."

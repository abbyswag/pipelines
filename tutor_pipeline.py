"""
Title: AI Teaching Function Pipeline
Author: EduSpheria
Description: A function-calling pipeline that teaches a domain using LLM-generated teaching steps and animated HTML visualizations.
Requirements: openai-compatible LLM, Open WebUI, html rendering frontend support
"""

import os
import uuid
import logging
from typing import Literal, List
from datetime import datetime
from pydantic import BaseModel, Field

from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Pipeline(FunctionCallingBlueprint):
    class Valves(FunctionCallingBlueprint.Valves):
        MODEL_NAME: str = Field(default="gpt-4", description="Model name for LLM")
        OUTPUT_DIR: str = Field(default="/app/pipelines/outputs", description="Where to store the generated HTMLs")
        MAX_STEPS: int = Field(default=5, description="Max number of learning steps")

    class Tools:
        def __init__(self, pipeline):
            self.pipeline = pipeline
            os.makedirs(self.pipeline.valves.OUTPUT_DIR, exist_ok=True)

        def generate_learning_outline(self, domain: str, steps: int = 5) -> List[dict]:
            prompt = f"""
You are an expert educator AI.
Generate a structured learning outline with {steps} steps to teach a beginner the topic: "{domain}".
Each step should contain:
- title
- short goal/visual_prompt (what the animation should explain)
Return the result as a JSON list.
"""
            llm = self.pipeline.get_llm()
            response = llm.chat.completions.create(
                model=self.pipeline.valves.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            import json
            return json.loads(response.choices[0].message.content)

        def generate_teaching_message(self, step_title: str) -> str:
            prompt = f"""
You are a teaching assistant. Write a detailed, engaging explanation for the learning step:
"{step_title}"
The output should sound like an expert tutor speaking directly to a student.
"""
            llm = self.pipeline.get_llm()
            response = llm.chat.completions.create(
                model=self.pipeline.valves.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()

        def generate_html_animation(self, step_title: str, visual_prompt: str) -> str:
            prompt = f"""
Create a beautiful, animated HTML page to help explain:
"{visual_prompt}"
Use advanced animations with Three.js, Anime.js, or Lottie.
Return complete HTML (self-contained).
"""
            llm = self.pipeline.get_llm()
            response = llm.chat.completions.create(
                model=self.pipeline.valves.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            html_code = response.choices[0].message.content.strip()
            filename = f"{step_title.replace(' ', '_').lower()}_{uuid.uuid4().hex[:6]}.html"
            filepath = os.path.join(self.pipeline.valves.OUTPUT_DIR, filename)
            with open(filepath, "w") as f:
                f.write(html_code)
            return f"sandbox://outputs/{filename}"

    def __init__(self):
        super().__init__()
        self.name = "AI Teaching Function Pipeline"
        self.valves = self.Valves(
            MODEL_NAME=os.getenv("MODEL_NAME", "gpt-4"),
            OUTPUT_DIR=os.getenv("OUTPUT_DIR", "/app/pipelines/outputs"),
            MAX_STEPS=int(os.getenv("MAX_STEPS", 5)),
            pipelines=["*"]
        )
        self.tools = self.Tools(self)

    def get_llm(self):
        from openai import OpenAI
        return OpenAI(base_url=os.getenv("VLLM_HOST", "http://localhost:8000/v1"), api_key=os.getenv("OPENAI_API_KEY", "sk-abc"))
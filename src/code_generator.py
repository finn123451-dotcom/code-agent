import openai
from typing import Optional
import os


class CodeGenerator:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_code(self, prompt: str, language: str = "python") -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a {language} programmer. Write clean, efficient code."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def generate_with_context(self, prompt: str, context: str, language: str = "python") -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a {language} programmer. Consider the context provided."},
                {"role": "user", "content": f"Context:\n{context}\n\nTask:\n{prompt}"}
            ]
        )
        return response.choices[0].message.content

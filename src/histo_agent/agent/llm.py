"""LLM client. Any OpenAI-compatible endpoint works (OpenAI, OpenRouter, Azure OpenAI, local vLLM)."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.0, max_tokens: int = 700) -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL") or None,
        temperature=temperature,
        max_tokens=max_tokens,
    )

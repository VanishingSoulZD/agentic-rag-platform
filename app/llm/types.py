from dataclasses import dataclass


@dataclass
class LLMResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    mock: bool

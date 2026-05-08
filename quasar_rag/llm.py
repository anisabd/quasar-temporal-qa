"""LLM loading and generation helpers (4-bit Llama-3.2-3B-Instruct)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from .config import LLM_MODEL_ID


def get_bnb_config() -> BitsAndBytesConfig:
    """Default 4-bit NF4 quantization config (T4 / consumer GPU friendly)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(model_id: str = LLM_MODEL_ID) -> Tuple[Any, Any]:
    """Load tokenizer + 4-bit quantized causal LM."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading quantized model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype:  {next(model.parameters()).dtype}")
    return model, tokenizer


def clean_model_reload() -> Tuple[Any, Any]:
    """Re-load the base model from scratch (clears any PEFT wrapping)."""
    torch.cuda.empty_cache()
    return load_model_and_tokenizer()


def build_pipeline(model, tokenizer):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def generate_response(
    pipe,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    top_k: int = 50,
    do_sample: bool = True,
) -> str:
    """Generate a single response for one chat-formatted message list."""
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        do_sample=do_sample,
    )
    return outputs[0]["generated_text"][-1]["content"]


def generate_all_responses_batch(
    pipe,
    prompts: List[List[Dict[str, str]]],
    batch_size: int = 8,
    max_new_tokens: int = 64,
    temperature: float = 0.3,
    top_p: float = 0.8,
    **generation_kwargs,
) -> List[List[str]]:
    """Run generation over a list of prompts using pipeline batching."""
    print(
        f"Generating responses for {len(prompts)} prompts in batches of {batch_size}"
    )
    all_responses: List[List[str]] = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        batch_outputs = pipe(
            batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **generation_kwargs,
        )
        for output in batch_outputs:
            messages = output[0]["generated_text"]
            last_msg = next(
                (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
                None,
            )
            all_responses.append([last_msg])
    print(f"Generated {len(all_responses)} responses")
    return all_responses

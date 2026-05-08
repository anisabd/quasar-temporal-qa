"""Q-LoRA fine-tuning pipeline (Section 5 of the report)."""

from __future__ import annotations

import os
from typing import Tuple

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .llm import get_bnb_config


# ----------------------------------------------------------------------
# Dataset preparation
# ----------------------------------------------------------------------
def prepare_training_data(
    dataset_dict: dict,
    prompt_manager,
    prompt_type: str = "standard_nl",
    response_type: str = "nl",
    num_shots: int = 0,
    split: str = "train",
) -> Dataset:
    """Convert (question, answer) pairs into HF chat-format training examples."""
    if response_type == "nl":
        row_name = "natural_lang_answers"
    elif response_type == "id":
        row_name = "original_answers"
    else:
        raise ValueError(f"Unknown response_type: {response_type}")

    train_df = dataset_dict["train"]
    split_df = dataset_dict[split]
    print(f"Preparing {len(split_df)} {split} examples...")

    training_examples = []
    for _, row in split_df.iterrows():
        query = row["question"]
        target_answer = row[row_name]

        messages = prompt_manager.create_prompt(
            query=query,
            prompt_type=prompt_type,
            num_shots=num_shots,
            train_df=train_df,
        )
        messages.append({"role": "assistant", "content": target_answer})

        training_examples.append(
            {
                "messages": messages,
                "question": query,
                "target_answer": target_answer,
            }
        )

    return Dataset.from_list(training_examples)


def format_chat_template(example, tokenizer):
    formatted_text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": formatted_text}


def tokenize_function(examples, tokenizer, max_length: int = 512):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ----------------------------------------------------------------------
# QLoRA configs
# ----------------------------------------------------------------------
def setup_qlora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )


def setup_training_args(output_dir: str = "./qlora-results") -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )


# ----------------------------------------------------------------------
# Fine-tune entry point
# ----------------------------------------------------------------------
def fine_tune_model(
    model,
    tokenizer,
    dataset_dict,
    prompt_manager,
    prompt_type: str = "standard_nl",
    response_type: str = "nl",
    num_shots: int = 0,
    output_dir: str = "./qlora-results",
):
    """Run the full fine-tuning pipeline; returns (model, trainer)."""
    print("Step 1: Setting up LoRA configuration...")
    lora_config = setup_qlora_config()
    model = prepare_model_for_kbit_training(model)

    print("Step 2: Applying LoRA to model...")
    model = get_peft_model(model, lora_config)

    print("Step 3: Preparing training dataset...")
    train_dataset = prepare_training_data(
        dataset_dict,
        prompt_manager,
        prompt_type=prompt_type,
        response_type=response_type,
        num_shots=num_shots,
        split="train",
    )
    eval_dataset = prepare_training_data(
        dataset_dict,
        prompt_manager,
        prompt_type=prompt_type,
        response_type=response_type,
        num_shots=num_shots,
        split="val",
    )

    print("Step 4: Formatting and tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: format_chat_template(x, tokenizer), batched=False
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_chat_template(x, tokenizer), batched=False
    )
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    print("Step 5: Setting up training arguments...")
    training_args = setup_training_args(output_dir=output_dir)

    print("Step 6: Creating trainer...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Step 7: Starting training...")
    try:
        checkpoints = (
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if os.path.exists(output_dir)
            else []
        )
        if checkpoints:
            latest = os.path.join(
                output_dir,
                sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1],
            )
            print(f"Resuming training from checkpoint: {latest}")
            trainer.train(resume_from_checkpoint=latest)
        else:
            print("No checkpoint found — training from scratch.")
            trainer.train()
    except ValueError as e:
        print(f"Could not resume from checkpoint ({e}). Starting from scratch.")
        trainer.train()

    print("Step 8: Saving model...")
    trainer.save_model()
    return model, trainer


# ----------------------------------------------------------------------
# Inference with adapter
# ----------------------------------------------------------------------
def load_finetuned_model(base_model_id: str, adapter_path: str) -> Tuple[AutoModelForCausalLM, None]:
    """Load the base 4-bit model and apply a saved Q-LoRA adapter."""
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    return PeftModel.from_pretrained(base_model, adapter_path)

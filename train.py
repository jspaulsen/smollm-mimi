import os
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
import wandb


load_dotenv()



def main(
    warmup_ratio: float = 0.05,
    lr_scheduler: str = "cosine",
    output_directory: Path = Path("runs-v4"),
) -> None:
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    learning_rate: float = 2e-4

    model_path: Path = Path("model")  # base model

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = load_dataset("jspaulsen/emilia-yodas-en-mimi-smollm2", split="train")

    # Filter out anything with a length > 2048 tokens
    dataset = dataset.filter(
        lambda x: len(x['input_ids']) <= 2048,
        num_proc=16,
    )

    split = dataset.train_test_split(test_size=0.00001)

    dataset = split["train"]
    eval_dataset = split["test"]

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)


    wandb.init(
        project="mimi-smollm-v3",
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": gradient_accumulation_steps,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": lr_scheduler,
        }
    )

    if not wandb.run or not wandb.run.name:
        raise ValueError("WandB run name is not set. Please set it before running the script.")

    name = wandb.run.name
    output_directory = output_directory / name
    output_directory.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")

    trainer = Trainer(
        model=model,
        train_dataset=dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        data_collator=data_collator,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            # warmup_steps=500,

            metric_for_best_model="loss",
            eval_strategy="steps",
            eval_steps=100,

            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_steps=1000,
            save_total_limit=10,

            # https://huggingface.co/docs/bitsandbytes/v0.43.0/en/optimizers#paged-optimizers
            optim="paged_adamw_8bit",
            # optim="adamw_8bit"
            weight_decay=0.01, # Turn this on if overfitting
            lr_scheduler_type=lr_scheduler,
            max_grad_norm=10.0,
            seed=3407,
            output_dir=str(output_directory),
            report_to="wandb",

            push_to_hub=True,
            hub_strategy="end"
        ),
    )

    trainer.args.set_push_to_hub("jspaulsen/smollm2-mimi")

    trainer.train()
    trainer.save_model("completed")


if __name__ == "__main__":
    main()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
import wandb


load_dotenv()



# <|soh|><|sot|>[text]<|eot|><|eoh|><|som|><|sos|>[speech]<|eos|><|eom|>

BOS = 1
SOH = 49152
EOH = 49153
SOT = 49154
EOT = 49155
SOM = 49156
EOM = 49157
SOS = 49158
EOS = 49159

# "<|eoh|>": 49153,
# "<|eom|>": 49157,
# "<|eos|>": 49159,
# "<|eot|>": 49155,
# "<|pad|>": 49160,
# "<|soh|>": 49152,
# "<|som|>": 49156,
# "<|sos|>": 49158,
# "<|sot|>": 49154


def transform_sample(
    tokenizer: AutoTokenizer,
    sample: dict,
) -> dict:
    query = sample['query']
    answer = sample['answer']
    ret = {}

    # Encode the query and answer
    query_ids = tokenizer(query, add_special_tokens=False)
    answer_ids = tokenizer(answer, add_special_tokens=False)

    # Create the input_ids and attention_mask
    input_ids = (
        [BOS, SOH, SOT] +
        query_ids['input_ids'] +
        [EOT, EOH, SOM, SOT] +
        answer_ids['input_ids'] +
        [EOT, EOM]
    )

    ret["input_ids"] = input_ids
    ret["labels"] = input_ids  # For training, labels are the same as input
    ret["attention_mask"] = [1] * len(input_ids)  # All tokens
    ret["length"] = len(input_ids)

    return ret


def main(
    epochs: int = 1,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.01,
    lr_scheduler: str = "cosine",
    batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    model_path: Path = Path("model"),
    output_directory: Path = Path("qa-runs"),
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    # https://huggingface.co/datasets/nampdn-ai/tiny-strange-textbooks
    # https://huggingface.co/datasets/roneneldan/TinyStories
    dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    dataset = dataset.map(
        lambda x: transform_sample(tokenizer, x),
        remove_columns=[column for column in dataset.column_names if column not in ['attention_mask', 'input_ids', 'length', 'labels']],
    )

    # Filter out any length > 6k
    dataset = dataset.filter(lambda x: x['length'] <= 2000)

    train_test = dataset.train_test_split(test_size=0.01, seed=42)
    dataset = train_test["train"]
    eval_dataset = train_test["test"]

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)


    wandb.init(
        project="smollm-mimi-qa-v2",
        # name=name,
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


    trainer = Trainer(
        model=model,
        train_dataset=dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_steps=100,
            save_total_limit=5,

            # Evaluation
            eval_steps=25,
            eval_strategy="steps",
            metric_for_best_model="loss",

            # https://huggingface.co/docs/bitsandbytes/v0.43.0/en/optimizers#paged-optimizers
            optim="paged_adamw_8bit",
            # optim="adamw_8bit",
            weight_decay=0.01, # Turn this on if overfitting
            lr_scheduler_type=lr_scheduler,

            max_grad_norm=10.0,
            # dataloader_drop_last=True,

            seed=3407,
            output_dir=str(output_directory),
            report_to="wandb",
        ),
    )

    trainer.train()
    # trainer.train(resume_from_checkpoint='qa-runs/desert-smoke-18/checkpoint-1000')


if __name__ == "__main__":
    main()

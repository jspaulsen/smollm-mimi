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


BOS = 1
SOH = 49152
EOH = 49153
SOT = 49154
EOT = 49155
SOM = 49156
EOM = 49157
SOS = 49158
EOS = 49159

AUDIO_CODE_TOKEN_OFFSET = 49161


def tokenize_audio(
    audio: list[int],
    starting_offset: int = AUDIO_CODE_TOKEN_OFFSET,
    codebook_size: int = 2048,
    codebooks: int = 8,
) -> list[int]:
    codes = []

    tensor = torch.tensor(audio, dtype=torch.int64)
    tensor = tensor.view(1, -1)
    tensor = tensor.view(1, 8, -1)

    assert len(audio) % codebooks == 0, "Audio must be a multiple of codebooks in length."
    assert isinstance(audio, list), "Audio must be a list of integers."

    for time in range(tensor.shape[2]):
        for codebook in range(tensor.shape[1]):
            token_id = tensor[0, codebook, time].item()
            offset = starting_offset + (codebook * codebook_size)
            codes.append(token_id + offset)

    # Assert at all codes are within the expected range
    return codes


def process_sample(
    tokenizer,
    sample: dict,
) -> dict:
    if 'codes' in sample:
        audio_codes = sample['codes']
    else:
        audio_codes: list[int] = sample['input_ids']

    text = sample['text']
    ret = {}

    # Tokenize the audio codes
    try:
        codes = tokenize_audio(audio_codes)
    except Exception as e:
        print(f"Error processing sample: {e}")
        return {}

    text_ids = tokenizer(text, add_special_tokens=False)
    text_ids = text_ids['input_ids']
    input_ids = (
        [BOS, SOH, SOT] +
        text_ids +
        [EOT, EOH, SOM, SOS] +
        codes +
        [EOS, EOM]
    )

    ret["input_ids"] = input_ids
    ret["labels"] = input_ids  # For training, labels are the same as input
    ret["attention_mask"] = [1] * len(input_ids)  # All tokens
    ret['length'] = len(codes)
    return ret



def main(
    warmup_ratio: float = 0.05,
    lr_scheduler: str = "cosine",
    output_directory: Path = Path("runs-v4"),
) -> None:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    learning_rate: float = 3e-4

    model_path: Path = Path("qa-runs/pious-fog-8/checkpoint-1400")  # text-qa
    # model_path: Path = Path("model")  # base model

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)


    def _process_sample(sample):
        return process_sample(tokenizer, sample)

    dataset = load_dataset("jspaulsen/emilia-en-mimi-small", split="train")
    dataset = dataset.map(
        _process_sample,
        remove_columns=[name for name in dataset.column_names if name not in ['input_ids', 'attention_mask', 'labels', 'length']],
        num_proc=8,  # Use 8 processes for faster processing
    )

    # Filter out anything with a length > 2048 tokens
    dataset = dataset.filter(lambda x: x['length'] <= 2048, num_proc=8)

    split = dataset.train_test_split(test_size=0.00001)

    dataset = split["train"]
    eval_dataset = split["test"]

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)


    wandb.init(
        project="mimi-smollm-v2",
        # name="radiant-bird-5",
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
            max_grad_norm=5.0,
            seed=3407,
            output_dir=str(output_directory),
            report_to="wandb",
        ),
    )

    trainer.train()
    # trainer.train(resume_from_checkpoint="runs-v4/feasible-river-10/checkpoint-37000")


if __name__ == "__main__":
    main()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

from datasets import Audio, load_dataset
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
import wandb

from src.mimi import MimiModel
from transform import process_sample


load_dotenv()



def main(
    epochs: int = 8,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.05,
    lr_scheduler: str = "cosine",
    batch_size: int = 6,
    gradient_accumulation_steps: int = 5,
    # model_path: Path = Path("model"),
    # model_path: Path = Path("runs/mimi-lfm2-v3-large/checkpoint-39339"),
    model_path: Path = Path("runs-v4/solar-shadow-11/checkpoint-66065"),
    output_directory: Path = Path("finetune"),
) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    encoder = MimiModel.from_pretrained(
        hf_repo="kyutai/tts-1.6b-en_fr",
        model_name="tokenizer-e351c8d8-checkpoint125.safetensors",
        device='cuda',
    )

    # Set the number of codebooks to 8
    encoder.set_num_codebooks(8)


    def _process_sample(sample):
        return process_sample(
            tokenizer, 
            encoder, 
            sample,
            transcript_key='text',
        )
    
    def filter_for_audio(sample):
        try:
            return len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"] <= 30.0
        except Exception as e:
            print(f"Error filtering audio for sample {sample['id']}: {e}")
            return False

    styles = [
        'default',
        'confused',
        'emphasis',
        'enunciated',
        'sad',
        # 'happy'
    ]

    # dataset = load_dataset(
    #     "ylacombe/expresso",
    #     split="train",
    #     # download_mode="force_redownload"
    # )

    dataset = load_dataset(
        "MrDragonFox/Elise",
        split="train",
    )

    # dataset = load_dataset(
    #     "audiofolder",
    #     data_dir="../dataset-redux/dataset/glimpsed/normalized",
    #     split="train",
    # )
    # dataset = dataset.filter(lambda x: x["speaker_id"] == "ex04") # filter for speaker_id ex04
    # dataset = dataset.filter(lambda x: x["style"] in styles)  # filter for styles
    # dataset = dataset.filter(lambda x: len(x["audio"]["array"]) / x["audio"]["sampling_rate"] <= 30.0)
    dataset = dataset.filter(filter_for_audio)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))


    # NOTE: This will filter out your entire dataset if you don't have the columns
    # filter out nisqa_mos	nisqa_noisiness	nisqa_discontinuity	nisqa_coloration if < 3.5
    # file_name,text,length_ms,p808_mos,mos_sig,mos_bak,mos_ovr,nisqa_mos,nisqa_noisiness,nisqa_discontinuity,nisqa_coloration,nisqa_loudness,ce,cu,pc,pq,sr_score,sr_prediction
    # dataset = dataset.filter(lambda x: x["nisqa_mos"] is not None and x["nisqa_mos"] >= 3.5)
    # dataset = dataset.filter(lambda x: x["nisqa_noisiness"] is not None and x["nisqa_noisiness"] >= 3.8)
    # dataset = dataset.filter(lambda x: x["nisqa_discontinuity"] is not None and x["nisqa_discontinuity"] >= 3.5)
    # dataset = dataset.filter(lambda x: x["nisqa_coloration"] is not None and x["nisqa_coloration"] >= 3.0)
    # dataset = dataset.filter(lambda x: x["sr_prediction"] == True)  # filter out any samples that have a sr_prediction of "False"

    dataset = dataset.map(_process_sample, remove_columns=[name for name in dataset.column_names if name not in ['input_ids', 'attention_mask', 'labels', 'length']])

    # Generate a split
    dataset_split = dataset.train_test_split(test_size=0.005)
    dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)


    wandb.init(
        project="mimi-lfm2-finetune-v2",
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
            # warmup_steps=500,

            metric_for_best_model="loss",
            eval_strategy="steps",
            eval_steps=25,

            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_steps=100,
            save_total_limit=10,

            # https://huggingface.co/docs/bitsandbytes/v0.43.0/en/optimizers#paged-optimizers
            optim="paged_adamw_8bit",
            # optim="adamw_8bit"
            weight_decay=0.01, # Turn this on if overfitting
            lr_scheduler_type=lr_scheduler,
            max_grad_norm=1.5,
            seed=3407,
            output_dir=str(output_directory),
            report_to="wandb",
        ),
    )

    trainer.train()


if __name__ == '__main__':
    main()

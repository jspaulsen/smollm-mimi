import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch

from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
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



def main() -> None:
    model_path: Path = Path("qa-runs/pious-fog-8/checkpoint-1400")  # text-qa
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    dataset = load_dataset("jspaulsen/emilia-yodas-en-mimi", split='train')
    dataset = dataset.map(
        lambda x: process_sample(tokenizer, x),
        remove_columns=[name for name in dataset.column_names if name not in ['input_ids', 'attention_mask', 'labels', 'length']],
        num_proc=8,  # Use 8 processes for faster processing
    )


    # push up the dataset to jspaulsen/emilia-yodas-en-mimi-smollm2
    dataset.push_to_hub("jspaulsen/emilia-yodas-en-mimi-smollm2")


if __name__ == '__main__':
    main()

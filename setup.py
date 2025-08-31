from pathlib import Path
from typing import cast
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



def main(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    codebooks: int = 8,
    codebook_size: int = 2048,
    output_directory: Path = Path("model"),
) -> None:
    """
    Setup the base model and tokenizer for training TTS utilizing Mimi.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # attn_implementation="flash_attention_2",
        device_map="auto",
        torch_dtype="auto",
        # torch_dtype="bfloat16",
    )

    audio_token_count = codebooks * codebook_size

    # <|soh|><|sot|>[text]<|eot|><|eoh|><|som|><|sos|>[speech]<|eos|><|eom|>
    # [text]<|speech_end|><|audio_start|>[audio_codes]<|audio_end|>
    audio_tokens = [f"<|audio_code_{i}|>" for i in range(audio_token_count)]
    special_tokens = {
        'soh': '<|soh|>',
        'eoh': '<|eoh|>',
        'sot': '<|sot|>',
        'eot': '<|eot|>',
        'som': '<|som|>',
        'eom': '<|eom|>',
        'sos': '<|sos|>',
        'eos': '<|eos|>',
        'pad': '<|pad|>'
    }

    # Add our new tokens to the tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
    tokenizer.add_tokens(audio_tokens)

    # If the pad token id isn't set, set it
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|pad|>"
        tokenizer.pad_token_id = tokenizer.get_vocab()["<|pad|>"]


    # Resize the model's token embeddings to match the new tokenizer size
    model.resize_token_embeddings(len(tokenizer))

    # Initialize audio token embeddings with structure awareness
    old_vocab_size = len(tokenizer) - len(audio_tokens)
    embeddings: Tensor = cast(Tensor, model.get_input_embeddings().weight.data)

    # Initialize each codebook's tokens similarly
    existing_mean = embeddings[:old_vocab_size].mean(dim=0)

    for codebook in range(8):
        start_idx = old_vocab_size + (codebook * codebook_size)
        end_idx = start_idx + codebook_size

        # Each codebook gets similar base + small variations
        base = existing_mean + torch.randn_like(existing_mean) * 0.01
        for i in range(start_idx, end_idx):
            embeddings[i] = base + torch.randn_like(existing_mean) * 0.005


    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    # Save the tokenizer and model
    tokenizer.save_pretrained(output_directory)
    model.save_pretrained(output_directory)


if __name__ == "__main__":
    main()

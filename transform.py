import torch

# from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from src.mimi import MimiModel


# This is the beginning offset for the audio code tokens.
# There are 2048 codes
AUDIO_CODE_TOKEN_OFFSET = 49160

# <|soh|><|sot|>[text]<|eot|><|eoh|><|som|><|sos|>[speech]<|eos|><|eom|>

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
# "<|soh|>": 49152,
# "<|som|>": 49156,
# "<|sos|>": 49158,
# "<|sot|>": 49154



def tokenize_audio(
    audio: torch.Tensor,  # [B, C, T]
) -> list[int]:
    codes = []

    assert audio.ndim == 3, "Audio tensor must be 3-dimensional."
    assert audio.shape[1] == 8, "Audio tensor must have 8 codebooks (channels)."

    for time in range(audio.shape[2]):
        for codebook in range(audio.shape[1]):
            token_id = audio[0, codebook, time].item()
            codes.append(token_id)

    return codes


def process_sample(
    tokenizer,
    encoder: MimiModel,
    sample: dict,
    # device: str = "cpu",
    transcript_key: str = "transcript",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    audio = sample["audio"]["array"]
    ret = {}

    # assert encoder.num_codebooks == codebooks, f"Encoder has {encoder.num_codebooks} codebooks, expected {codebooks}."
    # assert sample["audio"]["sampling_rate"] == 24000, "Audio must be resampled to 24000 Hz."

    audio = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension
    audio = audio.unsqueeze(0)  # Add channel dimension

    # Tokenize the audio codes
    result = encoder.encode(audio)
    codes = tokenize_audio(result)

    text_ids = tokenizer.encode(sample[transcript_key], add_special_tokens=False)
    input_ids = (
        [SOH, SOT] +
        text_ids +
        [EOT, EOH, SOM, SOS] +
        codes +
        [EOS, EOM]
    )

    ret["input_ids"] = input_ids
    ret["labels"] = input_ids  # For training, labels are the same as input
    ret["attention_mask"] = [1] * len(input_ids)  # All tokens
    return ret

import os

import torch
import safetensors.torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torchaudio

from src.mimi import MimiModel


AUDIO_CODE_TOKEN_OFFSET = 49161


def detokenize_audio(
    audio_tokens: list[int],
    starting_offset: int = AUDIO_CODE_TOKEN_OFFSET,
    codebook_size: int = 2048,
    num_codebooks: int = 8,
) -> torch.Tensor:
    ret = []

    assert len(audio_tokens) % num_codebooks == 0, f"Audio tokens must be divisible by {num_codebooks} codebooks"

    for n in range(0, len(audio_tokens), num_codebooks):
        batch_tokens = audio_tokens[n:n + num_codebooks]

        # Initialize output tensor for this batch [1, 8, 1]
        audio_step = torch.zeros((1, num_codebooks, 1), dtype=torch.int64)

        for codebook in range(num_codebooks):
            token_id = batch_tokens[codebook]
            offset = starting_offset + (codebook * codebook_size)
            original_code = token_id - offset
            audio_step[0, codebook, 0] = original_code

        ret.append(audio_step)
    return torch.cat(ret, dim=2)



def main() -> None:
    data = safetensors.torch.load_file("output.safetensors")["codes"]

    # convert tensor into a list of integers
    data = data.squeeze(0).tolist()

    encoder: MimiModel = MimiModel.from_pretrained(
        hf_repo="kyutai/tts-1.6b-en_fr",
        model_name="tokenizer-e351c8d8-checkpoint125.safetensors",
    )

    if len(data) % 8 != 0:
        data = data[:len(data) - (len(data) % 8)]

    audio_hat = detokenize_audio(data)
    audio = encoder.decode(audio_hat)

    print(audio.shape)  # Should print the shape of the tensor
    print(audio)  # Print the tensor to verify the output

    # Save it as a wav, 24k
    torchaudio.save("output.wav", audio.squeeze(0), 24000)
if __name__ == "__main__":
    main()

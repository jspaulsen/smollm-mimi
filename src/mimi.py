from __future__ import annotations
from pathlib import Path
from typing import Self

from huggingface_hub import hf_hub_download
from moshi.models.compression import MimiModel as _MimiModel
from moshi.models.loaders import _seanet_kwargs, _transformer_kwargs, _quantizer_kwargs, SAMPLE_RATE, FRAME_RATE, _is_safetensors
from moshi.modules import SEANetEncoder, SEANetDecoder, transformer
from moshi.quantization import SplitResidualVectorQuantizer
from safetensors.torch import load_model
import torch


class MimiModel(_MimiModel):
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        hf_repo: str | None = None,
        device: str = "cpu",
        num_codebooks: int = 32,
        **kwargs,
    ) -> Self:
        encoder = SEANetEncoder(**_seanet_kwargs)
        decoder = SEANetDecoder(**_seanet_kwargs)
        quantizer_kwargs = _quantizer_kwargs.copy()
        quantizer_kwargs["n_q"] = kwargs.get("n_q", num_codebooks)

        encoder_transformer = transformer.ProjectedTransformer(device=device, **_transformer_kwargs)
        decoder_transformer = transformer.ProjectedTransformer(device=device, **_transformer_kwargs)
        quantizer = SplitResidualVectorQuantizer(**quantizer_kwargs)

        ret = cls(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            channels=1,
            sample_rate=SAMPLE_RATE,
            frame_rate=FRAME_RATE,
            encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
            causal=True,
            resample_method="conv",
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
            **kwargs,  # Pass any additional kwargs to the base class constructor
        )

        ret = (
            ret
                .to(device=device)
                .eval()  # Set the model to evaluation mode
        )

        # TODO: Load the model weights and apply them
        if not hf_repo:
            fpath = Path(model_name)

            if not fpath.exists() or not fpath.is_file():
                raise FileNotFoundError(f"Model file not found: {fpath}")
        else:
            fpath = hf_hub_download(hf_repo, model_name)


        # TODO: Should we restore strict=True here?
        if _is_safetensors(fpath):
            missing, unexpected = load_model(ret, fpath, device=str(device), strict=False)

            if missing:
                print(f"Missing keys when loading model: {missing}")
        else:
            pkg = torch.load(fpath, "cpu")
            ret.load_state_dict(pkg["model"])

        ret.set_num_codebooks(num_codebooks)
        return ret

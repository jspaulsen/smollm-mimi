import os

import torch
import safetensors.torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


BOS = 1
SOH = 49152
EOH = 49153
SOT = 49154
EOT = 49155
SOM = 49156
EOM = 49157
SOS = 49158
EOS = 49159



def main(
    model_path: Path = Path("runs-v4/young-paper-12/checkpoint-13000"),
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    text = "What?"
    text_ids = tokenizer(text, add_special_tokens=False)
    input_values = text_ids["input_ids"]
    input_ids = [BOS, SOH, SOT] + input_values + [EOT, EOH, SOM, SOS]

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)  # Add batch dimension
    input_ids_tensor = input_ids_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

    output = model.generate(
        input_ids_tensor,
        do_sample=True,

        # temperature=0.3,
        # min_p=0.15,
        # repetition_penalty=1.05,

        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[EOM]
    )

    print(tokenizer.decode(output[0], skip_special_tokens=False))
    print(output)
    print(len(output[0]))

    # Find the first instance of SOS and add one.
    sos_index = output[0].tolist().index(SOS) # TODO: Revert once audio
    output = output[0][sos_index + 1:]


    # If the last token is EOM, remove it.
    if output[-1] == EOM:
        output = output[:-1]

    # if the last token is EOS
    if output[-1] == EOS:
        output = output[:-1]

    # Flatten the output tensor
    output = output.flatten()

    # and save as a safetensors file
    output_fpath = "output.safetensors"

    # Save the output to numpy file
    safetensors.torch.save_file({"codes": output}, output_fpath)


if __name__ == '__main__':
    main()

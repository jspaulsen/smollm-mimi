import datasets
from transformers import AutoTokenizer


# TODO: mimi.encode takes an argument, num_quantizers, which we should set to the number of codebooks (i.e., 8)


def main():
    tokenizer = AutoTokenizer.from_pretrained("model")
    dataset = datasets.load_dataset("jspaulsen/mls-eng-10k-lfm2-mimi", split="train")

    n = dataset[0]
    result = tokenizer.decode(n["input_ids"])
    print(result)


if __name__ == "__main__":
    main()

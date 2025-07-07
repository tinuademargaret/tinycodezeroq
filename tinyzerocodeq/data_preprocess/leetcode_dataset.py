#based on https://github.com/ganler/code-r1.git
import rich
import datasets
import json
import os
from shutil import copy
from os import makedirs
from rich.rule import Rule

SYSTEM_PROMPT = (
        "You are a helpful competitve programming problem generator."
        "The user will give you the solution to a programming problem. Your task is to generate a leetcode style programming word problem."
        "The generated problem must include a precise input and output format descriptions with at most one example of the input and output."
        "The generated problem must include the entry point class and function that is the main class and function of the given solution."
        "The generated problem must be a valid leetcode problem."
)

def leetcode2k():
    rich.print(Rule("Loading LeetCodeDataset..."))

    data_source = "newfacade/LeetCodeDataset"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            prompt = f"Please generate a leetcode style programming word problem for the following SOLUTION CODE\n. SOLUTION CODE:\n\n{example['completion'].strip()}. Don't forget to include the entry point class and function in the problem description."
            return {
                "data_source": "code",
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style":
                        "rule",
                    "ground_truth":
                        json.dumps({"functional": f"{example['test']}\n\ncheck({example['entry_point'].strip()})"}),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": example["query"],  # C++?
                    "prompt": prompt,
                    "dataset": "LeetCodeDataset",
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="tinyzerocodeq/data/leetcode")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset, test_dataset = leetcode2k()

    rich.print(Rule("Saving the final dataset"))
    rich.print(f"[bold green]Saving to {local_dir}...")

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

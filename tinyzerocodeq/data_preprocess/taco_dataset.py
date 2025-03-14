import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


# To extract the solution for each prompts in the dataset
def extract_solution(solution_str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/opt/tiger/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    num_few_shot = 5
    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Construct a `def make_map_fn(split)` for the corresponding datasets.
    # ...
    def make_map_fn():
        """
        extract solution
        add instruction
        reward model style -> rule
        pass input and ouput fields to reward model
        """
        pass

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    makedirs(hdfs_dir)

    copy(src=local_dir, dst=hdfs_dir)

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
    data_source = "BAAI/TACO"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        "Come up with a programming word problem for the following solution."
    )

    # Construct a `def make_map_fn(split)` for the corresponding datasets.
    # ...
    def make_map_fn(split):
        """
        extract solution
        add instruction
        reward model style -> rule
        pass input and ouput fields to reward model
        """

        def process_fn(example, idx):
            solution_raw = example.pop("solutions")[0]

            question = instruction_following + " " + solution_raw

            # The question is the answer
            answer_raw = example.pop("question")
            # solution = extract_solution(answer_raw)
            test_cases = example.pop("input_output")
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": test_cases},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": solution_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    makedirs(hdfs_dir)

    copy(src=local_dir, dst=hdfs_dir)

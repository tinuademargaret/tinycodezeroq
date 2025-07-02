# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import aiohttp
import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


async def single_compute_score(
    evaluation_func,
    completion,
    reference,
    task,
    task_extra_info,
    executor,
    timeout=300.0,
):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(
                        evaluation_func, task, completion, reference, task_extra_info
                    ),  # Ensure synchronous
                ),
                timeout=timeout,
            )
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    # except Exception as e:
    #     print(f"Error processing completion: {completion[:10]}, Error: {e}")
    #     return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        if extra_info is None:
            extra_info = [None] * len(tasks)
        # Create tasks for all rows

        tasks_async = [
            single_compute_score(
                evaluation_func,
                completion,
                reference,
                task,
                task_extra_info,
                executor,
                timeout=300.0,
            )
            for completion, reference, task, task_extra_info in zip(
                completions, references, tasks, extra_info
            )
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print("shut down failed: " + str(kill_err))
            raise

    # Process results
    for result, completion, reference, task in zip(
        results, completions, references, tasks
    ):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


async def single_inference(session, url, data):
    try:
        async with session.post(url, json=data) as r:
            response = await r.json()
            return response["content"]
    except Exception as e:
        print(f"Error in single inference: {e}")
        return None


async def parallel_inference(
    config,
    data,
):
    url = config.url
    timeout = aiohttp.ClientTimeout(
        total=config.timeout,
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        instruction = (
            "Given the problem description, write a complete solution in Python that adheres to the following guidelines:"
            "The solution must:"
            "- Be enclosed within a Python code block"
            "- Read the input from standard input (stdin) exactly as described in the problem statement"
            "- Process the input according to the problem's requirements"
            "- Output the result using the print() function exclusively (do not use return statements or stdout.write())"
        )
        task_async = [
            asyncio.create_task(
                single_inference(
                    session,
                    url,
                    {"prompt": instruction + "\n\n" + "PROBLEM: " + problem},
                )
            )
            for problem in data
        ]
        print(
            f"--------------------------------------NO OF TASKS: {len(task_async)}-------------------------------------------------------------------"
        )

        try:
            responses = await asyncio.gather(*task_async)
        except Exception as e:
            print(f"Error in parallel inference: {e}")
            responses = [None] * len(data)

        return responses


class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, config, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.config = config

    def verify(self, data):
        """
        solve generated problem (responses) to get solutions to verify
        verify the batch and save as ``acc`` tensor
        """

        generated_problem_ids = data.batch["responses"]
        generated_problem_str = self.tokenizer.batch_decode(
            generated_problem_ids, skip_special_tokens=True
        )

        generated_solution_ids = data.batch["solutions"]
        generated_solution_str = self.tokenizer.batch_decode(generated_solution_ids)

        # """
        # pass question_str to parallel_inference to get solutions
        # """
        # try:
        #     solution_str = asyncio.run(
        #         parallel_inference(self.config, generated_problem_str)
        #     )
        # except Exception as e:
        #     print(f"Error in parallel inference: {e}")
        #     solution_str = [None] * len(generated_problem_str)

        # batched scoring
        original_solution_ids = data.batch["prompts"]
        original_solution_str = self.tokenizer.batch_decode(
            original_solution_ids, skip_special_tokens=True
        )

        # ground truth is test cases
        ground_truth = [
            data_item.non_tensor_batch["reward_model"]["ground_truth"]
            for data_item in data
        ]
        data_sources = data.non_tensor_batch["data_source"]

        assert len(generated_solution_str) == len(ground_truth) == len(data_sources)
        print("COMPUTING SCORES.........")
        try:
            scores = asyncio.run(
                parallel_compute_score_async(
                    self.compute_score,
                    generated_solution_str,
                    ground_truth,
                    data_sources,
                    num_processes=64,
                )
            )
            print(f"SCORES: {scores}")
        except asyncio.TimeoutError as e:
            print("Global timeout in reward computing! Setting all as 0.")
            scores = [0.0 for _ in range(len(generated_solution_str))]
        # except Exception as e:
        #     print(
        #         f"Unexpected error in batched reward computing. Setting all as 0.: {e}"
        #     )
        #     scores = [0.0 for _ in range(len(solution_str))]
        data.batch["acc"] = torch.tensor(
            scores, dtype=torch.float32, device=original_solution_ids.device
        )
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        # this reward tensor is used to compute the advantages of the actor rollout i.e the generated problem
        # but the scores stored are from the  solution of the generated problem
        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )  # should be B, T

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(
            dim=-1
        )

        prompt_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        generated_problem_str = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        data_sources = data.non_tensor_batch["data_source"]
        extra_info = data.non_tensor_batch.get("extra_info", [None] * len(data_sources))

        scores = self.verify(data)  # should be B

        for i in range(len(data)):
            data_source = data_sources[i]
            # seems like we are storing scores at the last valid position
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(
                    f"------------------------------------------------------------PROMPT--------------------------------------------------------------------------------------"
                )
                print(prompt_str[i])
                print(
                    f"-------------------------------------------------------------Generated Problem---------------------------------------------------------------------------"
                )
                print({generated_problem_str[i]})

        return reward_tensor

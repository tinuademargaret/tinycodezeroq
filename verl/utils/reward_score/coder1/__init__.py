# evalution using firejail from https://github.com/ganler/code-r1.git

import re
from typing import Tuple, Optional
import time
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .utils import _ERROR_MSG_PREFIX
from .firejail_exec import code_exec_firejail as code_exec

_MAX_CHAR_DISPLAY = 2048

def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


def validate_response_structure(processed_str: str) -> bool:
    pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>$', re.DOTALL)
    return bool(pattern.match(processed_str.strip()))


# https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py
def try_extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if matches:
        final_answer = matches[-1].group(1).strip()
        return final_answer

    return solution_str


CODE_PATTERN = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)


def extract_code_from_string(solution_str):
    # solution_str = try_extract_solution(solution_str)
    code_blocks = CODE_PATTERN.findall(solution_str)
    return '\n'.join(code_blocks).strip()


def _compute_score(solution_str, ground_truth, extra_info, format_reward=0.1, answer_reward=1.):
    reward_log = []

    # ground_truth is not code, but tests
    # pass_fmt = validate_response_structure(solution_str)
    solution_code = extract_code_from_string(solution_str)
    print(f"SOLUTION CODE: {solution_code}")

    # if not pass_fmt or len(solution_code) == 0:  # only print full output when there is an error
    if len(solution_code) == 0:
        reward_log.append("-" * 16 + "Bad format detected!" + "-" * 16)
        reward_log.append("-" * 16 + "Original Model Output" + "-" * 16)
        reward_log.append(solution_str)
        return -answer_reward - format_reward, "\n".join(reward_log)

    reward_log.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)
    # print(f"GROUND TRUTH: {type(ground_truth)}")
    # ground_truth = json.loads(ground_truth)

    t_start = time.time()

    # log code
    if "functional" in ground_truth:
        reward_log.append(solution_code + "\n" + ground_truth["functional"])
    else:
        reward_log.append(solution_code)

    if "pytest" in ground_truth or "functional" in ground_truth:
        if "functional" in ground_truth:
            succ, output = code_exec(solution_code + "\n" + ground_truth["functional"])
        else:  # pytest
            succ, output = code_exec(solution_code, pytest=ground_truth["pytest"])
        if not succ:
            reward_log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
            reward_log.append(output[:_MAX_CHAR_DISPLAY])
            reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
            reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
            return format_reward, "\n".join(reward_log)
    elif "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list: str = ground_truth["inputs"]
        stdout_list: str = ground_truth["outputs"]

        # Add parallelism
        all_passed = True
        failed_output = None
        failed_stdin = None
        failed_stdout = None

        with ThreadPoolExecutor(max_workers=min(8, len(stdin_list))) as executor:
            futures = [
                executor.submit(remote_check_stdio, solution_code, stdin, stdout)
                for stdin, stdout in zip(stdin_list, stdout_list)
            ]
            for future in as_completed(futures): 
                succ, output, stdin, stdout = future.result()
            
                if succ is False or output.strip() != stdout.strip():
                    try:
                        all_passed = False
                        failed_output = output[:_MAX_CHAR_DISPLAY]  # truncate output to print
                        failed_stdin = stdin
                        failed_stdout = stdout
                        reward_log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
                        reward_log.append(f"🔎Input: {repr(failed_stdin)}")
                        reward_log.append(f"✅Expected: {repr(failed_stdout.strip())}")
                        reward_log.append(
                            f"❌Actual: {failed_output if failed_output.startswith(_ERROR_MSG_PREFIX) else repr(failed_output.strip())}")
                        reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                        reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
                        return format_reward, "\n".join(reward_log)
                    except Exception as e:
                        return format_reward, "\n".join(reward_log)
    else:
        raise ValueError(
            f"Current supports for ground-truth are ['functional', 'inputs/outputs'] -- No idea what's: {ground_truth = }"
        )

    # If we reach here, all tests passed
    reward_log.append("+" * 16 + "Test Execution Passed!" + "+" * 16)
    return format_reward + answer_reward, "\n".join(reward_log)


def compute_score(solution_str, ground_truth, extra_info, format_reward=0.1, answer_reward=1.):
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()
    score, reward_log = _compute_score(solution_str,
                                       ground_truth,
                                       extra_info=extra_info,
                                       format_reward=format_reward,
                                       answer_reward=answer_reward)
    
    marker = "✅" if score == (format_reward + answer_reward) else "❌"
    reward_log = marker * 16 + "Reward Calculation" + marker * 16 + "\n" + reward_log + "\n" + marker * 16 + f"Final Reward = {score}" + marker * 16
    # print(reward_log + "\n\n")
    return score, reward_log

#!/usr/bin/env python3

from simpledspy import pipe
import numpy as np
from math_calculator import MathCalculator
import random


calculator = MathCalculator()


def generate_task(num_digits):
    first_digit = np.random.randint(0, 10**num_digits)
    second_digit = np.random.randint(0, 10**num_digits)
    solution = first_digit * second_digit
    task = f"{first_digit} * {second_digit}"
    return task, solution

def sample_fewshot_samples(success_list):
    return_list = []
    # for sample in success_list:
    for i, sample in enumerate(success_list):
        # sample with prob of weight
        if random.random() < sample['weight']:
            sample['i'] = i
            return_list.append(sample)

    return return_list

def construct_prompt(fewshot_samples):
    prompt = ""
    for sample in fewshot_samples:
        prompt += f"Task: {sample['task']}\nOutput: {sample['output']}\n"
    return prompt

instruction = ''
metric_values = []
score_values = []
success_list = []
iteration = 0
while True:
# for i in range(10):
    task, solution = generate_task(4)
    fewshot_samples = sample_fewshot_samples(success_list)
    prompt = construct_prompt(fewshot_samples)
    input_ = f"{prompt}Task: {task}\nOutput: "
    reasoning, result = pipe(instruction, input_)
    output = (reasoning, result)
    # check if number 
    if result.isdigit() and int(result) == solution:
        # metric_values.append(1)
        metric_value = 1
        success_list.append({'task': task, 'output': output, 'metric': metric_value, 'weight': 1})
    else:
        # metric_values.append(0)
        metric_value = 0

    num_fewshot_samples = len(fewshot_samples)
    score = metric_value - (num_fewshot_samples/100)
    score_values.append(score)
    avg_score = np.mean(score_values[-100:])

    metric_values.append(metric_value)
    avg_metric = np.mean(metric_values[-100:])
    for sample in fewshot_samples:
        penalty_factor = 0.5
        # if metric_value == 1:
        if score > avg_score:
            success_list[sample['i']]['weight'] *= ((1-penalty_factor)/avg_metric) + penalty_factor
        else:
            success_list[sample['i']]['weight'] *= penalty_factor

        if success_list[sample['i']]['weight'] > 1:
            success_list[sample['i']]['weight'] = 1




    print(f"input_: {input_}")
    print(f"reasoning: {reasoning}")
    print(f"iter: {iteration}, task: {task}, Solution: {solution}, result: {result}, num_fs: {num_fewshot_samples}, Avg Metric: {avg_metric}")
    iteration += 1



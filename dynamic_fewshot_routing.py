#!/usr/bin/env python3

from simpledspy import pipe
import numpy as np
from math_calculator import MathCalculator
import random


calculator = MathCalculator()


def generate_multiplication_task(num_digits):
    first_digit = np.random.randint(0, 10**num_digits)
    second_digit = np.random.randint(0, 10**num_digits)
    solution = first_digit * second_digit
    task = f"{first_digit} * {second_digit}"
    return task, solution


def generate_program_reasoning_task():
    num_digits = 4
    first_digit = np.random.randint(0, 10**num_digits)
    second_digit = np.random.randint(0, 10**num_digits)
    if second_digit > 5000:
        solution = 10
    else:
        solution = first_digit * second_digit
        # return 100
    task = f"f({first_digit}, {second_digit}) = "
    return task, solution

def sample_memories(memory):
    return_list = []
    # for sample in memory:
    for i, sample in enumerate(memory):
        # sample with prob of weight
        if random.random() < sample['weight']:
            sample['i'] = i
            return_list.append(sample)

    return return_list

def construct_prompt(fewshot_samples):
    fewshot_str = ""
    hypothesis_str = ""
    for sample in fewshot_samples:
        if 'task' in sample:
            # fewshot_str += f"Task: {sample['task']}\nOutput: {sample['output']}\n"
            fewshot_str += f"Task: {sample['task']}\nOutput: {sample['output']}\Output score: {sample['metric']}\n"
        elif 'hypothesis' in sample:
            hypothesis_str += f"Hypothesis: {sample['hypothesis']}\n"

    return fewshot_str, hypothesis_str

instruction = ''
metric_values = []
score_values = []
memory = []
iteration = 0
while True:
# for i in range(10):
    task, solution = generate_multiplication_task(4)
    # task, solution = generate_program_reasoning_task()
    memory_samples = sample_memories(memory)
    # memory_str = construct_prompt(memory_samples)
    fewshot_str, hypothesis_str = construct_prompt(memory_samples)
    memory_str = f"Hypothesis: {hypothesis_str}\n Fewshot Examples:\n{fewshot_str}"
    input_ = f"Current task: {task}\n{memory_str}Task: {task}\nOutput: "
    reasoning, result, new_hypothesis = pipe(instruction, input_)
    output = (reasoning, result)
    # check if number 
    if result.isdigit() and int(result) == solution:
        # metric_values.append(1)
        # metric_value = 1
        metric_value = min(1, 1/abs(float(result) - solution + 0.00001))
    else:
        # metric_values.append(0)
        metric_value = 0

    num_memory_samples = len(memory_samples)
    score = metric_value - (num_memory_samples/100)
    score_values.append(score)
    avg_score = np.mean(score_values[-100:])

    metric_values.append(metric_value)
    avg_metric = np.mean(metric_values[-100:])
    num_fewshot_samples = len([sample for sample in memory_samples if 'task' in sample])
    num_hypothesis_samples = len([sample for sample in memory_samples if 'hypothesis' in sample])
    weight = avg_score if avg_score > 0 else 0.1
    memory.append({'hypothesis': new_hypothesis, 'weight': weight})
    memory.append({'task': task, 'output': output, 'metric': metric_value, 'weight': weight})
    # if metric_value == 1:
        # memory.append({'task': task, 'output': output, 'metric': metric_value, 'weight': weight})

    for sample in memory_samples:
        penalty_factor = 0.5
        # if metric_value == 1:
        if score > avg_score:
            memory[sample['i']]['weight'] *= ((1-penalty_factor)/avg_metric) + penalty_factor
        else:
            memory[sample['i']]['weight'] *= penalty_factor

        if memory[sample['i']]['weight'] > 1:
            memory[sample['i']]['weight'] = 1




    print(f"input_: {input_}")
    print(f"reasoning: {reasoning}")
    print(f"hypothesis_str: {hypothesis_str}")
    print(f'Hypothesis: {new_hypothesis}')
    print(f"iter: {iteration}, task: {task}, Solution: {solution}, result: {result}, num_fs: {num_fewshot_samples}, num_hypo: {num_hypothesis_samples}, Avg Metric: {avg_metric}")
    iteration += 1



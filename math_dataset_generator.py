import dspy
import math
import random
import json
from tqdm import tqdm

class MathTaskSignature(dspy.Signature):
    """Generate a math calculation task"""
    task_type = dspy.InputField(desc="Type of math operation to generate")
    task = dspy.OutputField(desc="A math calculation task as a string")

class MathDatasetGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_task = dspy.ChainOfThought(MathTaskSignature)
        
    def _generate_numbers(self, min_val=1, max_val=100):
        return random.randint(min_val, max_val), random.randint(min_val, max_val)

    def generate_dataset(self, num_tasks=100):
        operations = ['addition', 'subtraction', 'multiplication', 'division']
        dataset = []
        
        for _ in tqdm(range(num_tasks), desc="Generating Math Tasks"):
            op = random.choice(operations)
            a, b = self._generate_numbers()
            
            # Generate task description
            task_result = self.generate_task(task_type=op)
            task_str = task_result.task
            task_str = task_str.replace('a', str(a)).replace('b', str(b))
            
            # Calculate solution
            try:
                if op == 'addition':
                    solution = a + b
                elif op == 'subtraction':
                    solution = a - b
                elif op == 'multiplication':
                    solution = a * b
                elif op == 'division':
                    solution = round(a / b, 2)
                
                # Add to dataset if valid
                dataset.append({
                    'task': task_str,
                    'numbers': [a, b],
                    'operation': op,
                    'solution': solution
                })
            except Exception as e:
                continue
                
        return dataset

if __name__ == "__main__":
    # Configure DSPy
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.7, cache=False)
    dspy.settings.configure(lm=lm)
    
    # Generate dataset
    generator = MathDatasetGenerator()
    dataset = generator.generate_dataset(num_tasks=1000)
    
    # Save to file
    with open("math_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} math tasks. Saved to math_dataset.json")

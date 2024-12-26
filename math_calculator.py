import dspy
import json
from tqdm import tqdm

class MathCalculationSignature(dspy.Signature):
    """Solve math calculation tasks using chain-of-thought reasoning"""
    task = dspy.InputField(desc="The math calculation task to solve")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning to solve the task")
    solution = dspy.OutputField(desc="The numerical solution to the task")

class MathCalculator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.calculate = dspy.ChainOfThought(MathCalculationSignature)

    def evaluate_on_dataset(self, dataset_path="math_dataset.json"):
        with open(dataset_path) as f:
            dataset = json.load(f)
        
        correct = 0
        total = len(dataset)
        
        for item in tqdm(dataset, desc="Evaluating Math Calculator"):
            task = item['task']
            expected_solution = item['solution']
            
            result = self.calculate(task=task)
            
            try:
                # Compare solutions with some tolerance for floating point
                if abs(float(result.solution) - float(expected_solution)) < 0.01:
                    correct += 1
            except:
                continue
                
        accuracy = correct / total
        print(f"Math Calculator Accuracy: {accuracy:.1%}")
        return accuracy

if __name__ == "__main__":
    # Configure DSPy
    lm = dspy.LM(model="deepseek/deepseek-chat", temperature=0.3, cache=False)
    dspy.settings.configure(lm=lm)
    
    # Train and evaluate calculator
    calculator = MathCalculator()
    calculator.evaluate_on_dataset()

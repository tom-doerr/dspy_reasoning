#!/usr/bin/env python3
import dspy

# Step 1: Configure the LM to use DeepSeek with temperature=1 and no caching
lm = dspy.LM(model="deepseek/deepseek-chat", temperature=1, cache=False)  # Use DeepSeek as the LM
dspy.settings.configure(lm=lm)


action_list = ['reasoning', 'terminate']
# Step 2: Define the Signature for Action-Oriented Reasoning
class ReasoningSignature(dspy.Signature):
    context = dspy.InputField(desc="The context to reason about")
    objective = dspy.InputField(desc="The objective to achieve")
    reasoning_output = dspy.OutputField(desc="The reasoning process")
    # action = dspy.OutputField(desc="The action to take, must be either 'reasoning' or 'terminate'")
    action = dspy.OutputField(format=action_list, desc="The action to take, must be either 'reasoning' or 'terminate'")

# Step 3: Create a Module with the Signature
class ActionReasoning(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought for action-oriented reasoning
        self.generate_action = dspy.ChainOfThought(ReasoningSignature)

    def forward(self, context, objective):
        # Run the reasoning pipeline
        return self.generate_action(context=context, objective=objective)

# Step 4: Create an Instance of the Pipeline
reasoning_pipeline = ActionReasoning()

def run_reasoning_pipeline(initial_context, initial_objective, callback=None):
    # Extract just the question and hint from the initial context
    context_lines = initial_context.strip().split('\n')
    question = next(line for line in context_lines if line.startswith("Final Question:"))
    hint = next(line for line in context_lines if line.startswith("Hint:"))
    
    context = f"{question}\n{hint}"
    objective = initial_objective
    iteration = 1
    
    while True:
        print(f"\n--- Reasoning Iteration {iteration} ---")
        print(f"Question: {question}")
        print(f"Hint: {hint}")
        print(f"Objective: {objective}")
        
        # Run the reasoning pipeline
        result = reasoning_pipeline(context=context, objective=objective)
        
        # Call callback if provided
        if callback:
            callback(iteration, context, objective, result)
        
        # Validate and process the action
        action = result.action.lower().strip()
        print("action:", action)
        if "terminate" in action or "no further" in action:
            print("\nFinal Reasoning Output:", result.reasoning_output)
            # print("Decision: Terminate reasoning process")
            break
            
        print("Reasoning Output:", result.reasoning_output)
        # print("Decision: Continue reasoning")
        action = result.action.lower().strip()
        
        # Update context and objective for next iteration
        context = result.reasoning_output
        objective = "Continue reasoning based on previous analysis"
        iteration += 1

# Example usage
initial_context = "The Eiffel Tower is located in Paris, France. It was completed in 1889 and is one of the most famous landmarks in the world."
initial_objective = "Analyze the historical significance of the Eiffel Tower and determine if further investigation is needed"

if __name__ == "__main__":
    run_reasoning_pipeline(initial_context, initial_objective)

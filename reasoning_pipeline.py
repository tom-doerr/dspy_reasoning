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
    reasoning = dspy.OutputField(desc="The reasoning process including step-by-step calculations")
    reasoning_output = dspy.OutputField(desc="The final output of the reasoning process")
    is_valid_reasoning = dspy.OutputField(
        desc="True only if the reasoning includes correct mathematical calculations that reach the target value"
    )
    action = dspy.OutputField(
        desc="The action to take, must be either 'reasoning' or 'terminate'"
    )

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
    # Use the full context initially
    context = initial_context.strip()
    # Extract question and hint if they exist
    context_lines = context.split('\n')
    question = next((line for line in context_lines if line.startswith("Final Question:")), context)
    hint = next((line for line in context_lines if line.startswith("Hint:")), "")
    
    # Create display context for debugging
    display_context = f"{question}\n{hint}" if hint else question
    objective = initial_objective
    iteration = 1
    
    while True:
        print(f"\n--- Reasoning Iteration {iteration} ---")
        print(f"Context: {display_context}")
        print(f"Objective: {objective}")
        
        # Run the reasoning pipeline
        result = reasoning_pipeline(context=context, objective=objective)
        
        # Call callback if provided
        if callback:
            callback(iteration, context, objective, result)
        
        # Validate and process the action
        action = result.action.lower().strip()
        print("action:", action)
        print("Reasoning Process:", result.reasoning)
        print("Reasoning Output:", result.reasoning_output)
        print("Is Valid Reasoning:", result.is_valid_reasoning)
        
        # Only accept termination if the reasoning is mathematically valid
        if ("terminate" in action or "no further" in action) and result.is_valid_reasoning:
            print("Decision: Terminate reasoning process with valid solution")
            break
        elif ("terminate" in action or "no further" in action) and not result.is_valid_reasoning:
            print("Decision: Invalid solution found - continuing reasoning")
            objective = "The previous solution was mathematically incorrect. Try a different approach."
            continue
            
        print("Decision: Continue reasoning")
        
        # Update context and objective for next iteration
        context = result.reasoning_output
        objective = "Continue reasoning based on previous analysis"
        iteration += 1

# Example usage
initial_context = """
How can you solve the Game of 24 using the numbers 3,
4, 5, and 6?
Let's think step by step:
1. We need to use basic arithmetic operations (+, -, *, /) to 
get 24.
2. One possible solution is: (3 * 4) + (5 + 6) = 24."""
initial_objective = "Generate a new solution using the same numbers."

if __name__ == "__main__":
    run_reasoning_pipeline(initial_context, initial_objective)

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
    
    # Formal logical fallacy detection
    affirming_consequent_analysis = dspy.OutputField(
        desc="Analysis of whether the fallacy of affirming the consequent was committed"
    )
    affirming_consequent_confidence = dspy.OutputField(
        desc="Confidence score from 1-10 where 1 means extremely sure no fallacy was committed and 10 means fallacy was definitely committed"
    )
    
    denying_antecedent_analysis = dspy.OutputField(
        desc="Analysis of whether the fallacy of denying the antecedent was committed"
    )
    denying_antecedent_confidence = dspy.OutputField(
        desc="Confidence score from 1-10 where 1 means extremely sure no fallacy was committed and 10 means fallacy was definitely committed"
    )
    
    undistributed_middle_analysis = dspy.OutputField(
        desc="Analysis of whether the fallacy of the undistributed middle was committed"
    )
    undistributed_middle_confidence = dspy.OutputField(
        desc="Confidence score from 1-10 where 1 means extremely sure no fallacy was committed and 10 means fallacy was definitely committed"
    )
    
    illicit_major_analysis = dspy.OutputField(
        desc="Analysis of whether the fallacy of illicit major was committed"
    )
    illicit_major_confidence = dspy.OutputField(
        desc="Confidence score from 1-10 where 1 means extremely sure no fallacy was committed and 10 means fallacy was definitely committed"
    )
    
    illicit_minor_analysis = dspy.OutputField(
        desc="Analysis of whether the fallacy of illicit minor was committed"
    )
    illicit_minor_confidence = dspy.OutputField(
        desc="Confidence score from 1-10 where 1 means extremely sure no fallacy was committed and 10 means fallacy was definitely committed"
    )
    
    is_valid_reasoning = dspy.OutputField(
        desc="True if the reasoning is mathematically valid and reaches the correct conclusion"
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
    # Initialize context history
    context_history = [initial_context.strip()]
    
    # Extract question and hint if they exist
    context_lines = initial_context.split('\n')
    question = next((line for line in context_lines if line.startswith("Final Question:")), initial_context)
    hint = next((line for line in context_lines if line.startswith("Hint:")), "")
    
    # Create display context for debugging
    display_context = f"{question}\n{hint}" if hint else question
    objective = initial_objective
    iteration = 1
    
    while True:
        print(f"\n--- Reasoning Iteration {iteration} ---")
        print(f"Context: {display_context}")
        print(f"Objective: {objective}")
        
        # Get current context from history
        current_context = "\n\n".join(context_history)
        
        # Run the reasoning pipeline
        result = reasoning_pipeline(context=current_context, objective=objective)
        
        # Call callback if provided
        if callback:
            callback(iteration, current_context, objective, result)
        
        # Validate and process the action
        action = result.action.lower().strip()
        print("Reasoning Process:", result.reasoning)
        print("Reasoning Output:", result.reasoning_output)
        print("\nFormal Logical Fallacy Analysis:")
        print(f"Affirming Consequent: {result.affirming_consequent_analysis} (Confidence: {result.affirming_consequent_confidence}/10)")
        print(f"Denying Antecedent: {result.denying_antecedent_analysis} (Confidence: {result.denying_antecedent_confidence}/10)")
        print(f"Undistributed Middle: {result.undistributed_middle_analysis} (Confidence: {result.undistributed_middle_confidence}/10)")
        print(f"Illicit Major: {result.illicit_major_analysis} (Confidence: {result.illicit_major_confidence}/10)")
        print(f"Illicit Minor: {result.illicit_minor_analysis} (Confidence: {result.illicit_minor_confidence}/10)")
        print("action:", action)
        
        # Only accept termination if explicitly told to
        if "terminate" in action or "no further" in action:
            if result.is_valid_reasoning.lower().strip() in ["true", "yes", "correct"]:
                print("Decision: Terminate reasoning process with valid solution")
                break
            else:
                print("Decision: Invalid solution found - continuing reasoning")
                objective = "The previous solution was mathematically incorrect. Try a different approach."
                continue
            
        print("Decision: Continue reasoning")
        
        # Update context history with full reasoning details
        context_history.append(f"""
--- Reasoning Iteration {iteration} ---
Context: {display_context}
Objective: {objective}
Reasoning Process: {result.reasoning}
Reasoning Output: {result.reasoning_output}
Fallacy Analysis:
- Affirming Consequent: {result.affirming_consequent_analysis} (Confidence: {result.affirming_consequent_confidence}/10)
- Denying Antecedent: {result.denying_antecedent_analysis} (Confidence: {result.denying_antecedent_confidence}/10)
- Undistributed Middle: {result.undistributed_middle_analysis} (Confidence: {result.undistributed_middle_confidence}/10)
- Illicit Major: {result.illicit_major_analysis} (Confidence: {result.illicit_major_confidence}/10)
- Illicit Minor: {result.illicit_minor_analysis} (Confidence: {result.illicit_minor_confidence}/10)
""".strip())
        
        # Update context for next iteration with full history
        context = "\n\n".join(context_history)
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

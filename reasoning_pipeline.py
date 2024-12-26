#!/usr/bin/env python3
import dspy

# Step 1: Configure the LM to use DeepSeek with temperature=1 and no caching
lm = dspy.LM(model="deepseek/deepseek-chat", temperature=1, cache=False)  # Use DeepSeek as the LM
dspy.settings.configure(lm=lm)


action_list = ['reasoning', 'terminate']
# Step 2: Define the Signature for Core Reasoning
class ReasoningSignature(dspy.Signature):
    context = dspy.InputField(desc="The context to reason about")
    objective = dspy.InputField(desc="The objective to achieve")
    reasoning = dspy.OutputField(desc="The reasoning process including step-by-step calculations")
    reasoning_output = dspy.OutputField(desc="The final output of the reasoning process")
    
    informal_proof = dspy.OutputField(
        desc="A detailed informal proof written in natural language that explains the reasoning step-by-step in a clear and accessible way, including assumptions, logical connections, and conclusions"
    )
    
    is_valid_reasoning = dspy.OutputField(
        desc="True if the reasoning is mathematically valid and reaches the correct conclusion"
    )
    
    action = dspy.OutputField(
        desc="The action to take, must be either 'reasoning' or 'terminate'"
    )

# Define Signature for Analysis
class ReasoningAnalysisSignature(dspy.Signature):
    context = dspy.InputField(desc="The context of the reasoning")
    reasoning = dspy.InputField(desc="The reasoning process to analyze")
    reasoning_output = dspy.InputField(desc="The output of the reasoning process")
    
    objective_achieved_analysis = dspy.OutputField(
        desc="Analysis of whether the objective was fully achieved"
    )
    
    objective_achieved_confidence = dspy.OutputField(
        desc="Confidence score from 1-10 where 1 means extremely sure objective was not achieved and 10 means objective was definitely achieved"
    )
    

# Step 3: Create a Module with the Signature
class ActionReasoning(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought for core reasoning
        self.generate_action = dspy.ChainOfThought(ReasoningSignature)
        # Separate module for analysis
        self.analyze_reasoning = dspy.ChainOfThought(ReasoningAnalysisSignature)

    def forward(self, context, objective):
        # First generate the reasoning
        reasoning_result = self.generate_action(context=context, objective=objective)
        
        # Then analyze the reasoning
        analysis_result = self.analyze_reasoning(
            context=context,
            reasoning=reasoning_result.reasoning,
            reasoning_output=reasoning_result.reasoning_output
        )
        
        # Combine results safely, ensuring all required fields are present
        combined = {
            "reasoning": reasoning_result.reasoning,
            "reasoning_output": reasoning_result.reasoning_output,
            "informal_proof": reasoning_result.informal_proof,
            "is_valid_reasoning": reasoning_result.is_valid_reasoning,
            "action": reasoning_result.action,
            "objective_achieved_analysis": analysis_result.objective_achieved_analysis,
            "objective_achieved_confidence": analysis_result.objective_achieved_confidence,
            "affirming_consequent_analysis": analysis_result.affirming_consequent_analysis,
            "affirming_consequent_confidence": analysis_result.affirming_consequent_confidence,
            "denying_antecedent_analysis": analysis_result.denying_antecedent_analysis,
            "denying_antecedent_confidence": analysis_result.denying_antecedent_confidence,
            "undistributed_middle_analysis": analysis_result.undistributed_middle_analysis,
            "undistributed_middle_confidence": analysis_result.undistributed_middle_confidence,
            "illicit_major_analysis": analysis_result.illicit_major_analysis,
            "illicit_major_confidence": analysis_result.illicit_major_confidence,
            "illicit_minor_analysis": analysis_result.illicit_minor_analysis,
            "illicit_minor_confidence": analysis_result.illicit_minor_confidence
        }
        return dspy.Prediction(**combined)

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
        print("\nDetailed Informal Proof:")
        print(result.informal_proof)
        print("\nObjective Achievement Analysis:")
        print(f"{result.objective_achieved_analysis} (Confidence: {result.objective_achieved_confidence}/10)")
        
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

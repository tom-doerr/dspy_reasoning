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
    reasoning = dspy.OutputField(desc="The reasoning process including step-by-step calculations", optional=True)
    reasoning_output = dspy.OutputField(desc="The final output of the reasoning process", optional=True)
    
    informal_proof = dspy.OutputField(
        desc="A detailed informal proof written in natural language that explains the reasoning step-by-step in a clear and accessible way, including assumptions, logical connections, and conclusions",
        optional=True
    )

# Define Signature for Analysis
class RequirementsSignature(dspy.Signature):
    context = dspy.InputField(desc="The context of the reasoning")
    objective = dspy.InputField(desc="The objective to achieve")
    current_requirements = dspy.InputField(desc="List of current requirements to achieve the objective")
    new_requirements = dspy.OutputField(desc="List of new requirements to add to achieve the objective")
    unnecessary_requirements = dspy.OutputField(desc="List of requirements that are no longer needed to achieve the objective")
    action = dspy.OutputField(desc="The action to take, must be either 'add_requirements', 'remove_requirements', or 'stop'")

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
    
    is_valid_reasoning = dspy.OutputField(
        desc="True if the reasoning in the input is valid and reaches the correct conclusion"
    )
    
    action = dspy.OutputField(
        desc="The action to take, must be either 'reasoning' or 'terminate'"
    )
    

# Step 3: Create a Module with the Signature
class RequirementsGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_requirements = dspy.ChainOfThought(RequirementsSignature)

    def forward(self, context, objective, current_requirements):
        result = self.generate_requirements(
            context=context,
            objective=objective,
            current_requirements=current_requirements
        )
        return result

class ActionReasoning(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought for core reasoning
        self.generate_action = dspy.ChainOfThought(ReasoningSignature)
        # Separate module for analysis
        self.analyze_reasoning = dspy.ChainOfThought(ReasoningAnalysisSignature)
        # Module for requirements generation
        self.requirements_generator = RequirementsGenerator()

    def forward(self, context, objective):
        # First generate the reasoning
        reasoning_result = self.generate_action(context=context, objective=objective)
        
        # Then analyze the reasoning
        analysis_result = self.analyze_reasoning(
            context=context,
            reasoning=reasoning_result.reasoning,
            reasoning_output=reasoning_result.reasoning_output
        )
        
        # Combine results safely with fallback values
        combined = {
            "reasoning": getattr(reasoning_result, "reasoning", "No reasoning provided"),
            "reasoning_output": getattr(reasoning_result, "reasoning_output", "No output provided"),
            "informal_proof": getattr(reasoning_result, "informal_proof", "No proof provided"),
            "objective_achieved_analysis": analysis_result.objective_achieved_analysis,
            "objective_achieved_confidence": analysis_result.objective_achieved_confidence,
            "is_valid_reasoning": analysis_result.is_valid_reasoning,
            "action": analysis_result.action
        }
        return dspy.Prediction(**combined)

# Step 4: Create an Instance of the Pipeline
reasoning_pipeline = ActionReasoning()

def generate_requirements(context, objective):
    """Iteratively generate and refine requirements for achieving an objective"""
    requirements = []
    iteration = 1
    
    while True:
        print(f"\n--- Requirements Iteration {iteration} ---")
        print(f"Current Requirements: {requirements}")
        
        # Generate new requirements
        result = RequirementsGenerator()(
            context=context,
            objective=objective,
            current_requirements=requirements
        )
        
        # Process new requirements
        if result.new_requirements:
            print(f"Adding new requirements: {result.new_requirements}")
            requirements.extend(result.new_requirements)
        
        # Process unnecessary requirements
        if result.unnecessary_requirements:
            print(f"Removing unnecessary requirements: {result.unnecessary_requirements}")
            requirements = [r for r in requirements if r not in result.unnecessary_requirements]
        
        # Check if we should stop
        if result.action.lower().strip() == "stop":
            print("Requirements generation complete")
            break
            
        iteration += 1
    
    return requirements

def run_reasoning_pipeline(initial_context, initial_objective, callback=None):
    # Generate requirements first
    requirements = generate_requirements(initial_context, initial_objective)
    print(f"\nFinal Requirements: {requirements}")
    
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
Objective Analysis: {result.objective_achieved_analysis} (Confidence: {result.objective_achieved_confidence}/10)
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

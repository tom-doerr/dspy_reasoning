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
    reasoning_output = dspy.OutputField(
        desc="The final output of the reasoning process. If no specific output, repeat the reasoning conclusion.",
        optional=True
    )
    informal_proof = dspy.OutputField(
        desc="A numbered list of steps for the informal proof. If no proof needed, summarize the reasoning steps.",
        optional=True
    )

# Define Signature for Analysis
class RequirementsSignature(dspy.Signature):
    context = dspy.InputField(desc="The context of the reasoning")
    objective = dspy.InputField(desc="The objective to achieve")
    current_requirements = dspy.InputField(desc="List of current requirements to achieve the objective")
    new_requirements = dspy.OutputField(
        desc="List of new requirements to add to achieve the objective. Return an empty list if no new requirements are needed.",
        default=[]
    )
    unnecessary_requirements = dspy.OutputField(
        desc="List of requirements that are no longer needed to achieve the objective. Return an empty list if no requirements should be removed.",
        default=[]
    )
    action = dspy.OutputField(
        desc="The action to take: 'add_requirements' if new requirements are needed, 'remove_requirements' if requirements should be removed, or 'stop' if requirements are complete",
        default="stop"
    )

class ReasoningAnalysisSignature(dspy.Signature):
    context = dspy.InputField(desc="The context of the reasoning")
    reasoning = dspy.InputField(desc="The reasoning process to analyze")
    reasoning_output = dspy.InputField(desc="The output of the reasoning process")
    informal_proof = dspy.InputField(desc="The numbered list of proof steps to analyze")
    
    proof_line_analysis = dspy.OutputField(
        desc="Detailed analysis of each proof line, checking if it makes logical sense and is mathematically correct"
    )
    
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
        
        # Handle missing fields
        reasoning = getattr(reasoning_result, "reasoning", "No reasoning provided")
        reasoning_output = getattr(reasoning_result, "reasoning_output", reasoning)
        informal_proof = getattr(reasoning_result, "informal_proof", reasoning)
        
        # Then analyze the reasoning and proof
        analysis_result = self.analyze_reasoning(
            context=context,
            reasoning=reasoning,
            reasoning_output=reasoning_output,
            informal_proof=informal_proof
        )
        
        # Handle missing analysis fields
        objective_achieved_analysis = getattr(analysis_result, "objective_achieved_analysis", "No analysis provided")
        objective_achieved_confidence = getattr(analysis_result, "objective_achieved_confidence", 5)
        is_valid_reasoning = getattr(analysis_result, "is_valid_reasoning", "unknown")
        action = getattr(analysis_result, "action", "reasoning")
        proof_line_analysis = getattr(analysis_result, "proof_line_analysis", "No proof line analysis provided")
        
        combined = {
            "reasoning": reasoning,
            "reasoning_output": reasoning_output,
            "informal_proof": informal_proof,
            "objective_achieved_analysis": objective_achieved_analysis,
            "objective_achieved_confidence": objective_achieved_confidence,
            "is_valid_reasoning": is_valid_reasoning,
            "action": action,
            "proof_line_analysis": proof_line_analysis
        }
        return dspy.Prediction(**combined)

# Step 4: Create an Instance of the Pipeline
reasoning_pipeline = ActionReasoning()

def generate_requirements(context, objective):
    """Iteratively generate and refine requirements for achieving an objective"""
    requirements = []
    iteration = 1
    max_iterations = 10
    
    while iteration <= max_iterations:
        # If we hit max iterations, reset completely and try again
        if iteration == max_iterations:
            print("\nWarning: Reached max iterations. Resetting requirements and starting fresh.")
            requirements = []
            iteration = 1
            continue
            
        print(f"\n--- Requirements Iteration {iteration} ---")
        print("Current Requirements:")
        for i, req in enumerate(requirements, 1):
            print(f"{i}. {req}")
        
        # Generate new requirements
        result = RequirementsGenerator()(
            context=context,
            objective=objective,
            current_requirements=requirements
        )
        
        # Process new requirements
        if result.new_requirements:
            if isinstance(result.new_requirements, str):
                # Split string into list items and filter out non-requirement statements
                new_reqs = [
                    req.strip() for req in result.new_requirements.split('\n') 
                    if req.strip() and not req.lower().startswith(('none', 'no new'))
                ]
            else:
                # Filter list items for non-requirement statements
                new_reqs = [
                    req for req in result.new_requirements 
                    if not str(req).lower().startswith(('none', 'no new'))
                ]
                
            if new_reqs:  # Only add if we have actual requirements
                print("\nAdding new requirements:")
                for req in new_reqs:
                    print(f"- {req}")
                    requirements.append(req)
        
        # Process unnecessary requirements
        if result.unnecessary_requirements:
            if isinstance(result.unnecessary_requirements, str):
                # Split string into list items
                remove_reqs = [req.strip() for req in result.unnecessary_requirements.split('\n') if req.strip()]
            else:
                remove_reqs = result.unnecessary_requirements
                
            print("\nRemoving unnecessary requirements:")
            for req in remove_reqs:
                print(f"- {req}")
                requirements = [r for r in requirements if r not in remove_reqs]
        
        # Check if we should stop
        if result.action.lower().strip() == "stop":
            print("\nRequirements generation complete")
            break
            
        iteration += 1
    
    print("\nFinal Requirements:")
    for i, req in enumerate(requirements, 1):
        print(f"{i}. {req}")
    
    return requirements

def track_analysis(analysis_history, analysis, confidence):
    """Track analysis results as a list of tuples"""
    try:
        # Extract first digit if confidence is a string
        if isinstance(confidence, str):
            confidence = ''.join(filter(str.isdigit, confidence)) or '5'
        confidence_int = int(confidence)
        # Clamp to 1-10 range
        confidence_int = max(1, min(10, confidence_int))
        analysis_history.append((analysis, confidence_int))
    except (ValueError, TypeError):
        # Default to medium confidence if parsing fails
        analysis_history.append((analysis, 5))
    return analysis_history

def run_reasoning_pipeline(initial_context, initial_objective, callback=None):
    # Generate requirements first with retry logic
    max_retries = 3
    requirements = []
    
    for attempt in range(max_retries):
        requirements = generate_requirements(initial_context, initial_objective)
        print(f"\nFinal Requirements: {requirements}")
        
        # If we got requirements, break
        if requirements:
            break
            
        print(f"\nWarning: Empty requirements list on attempt {attempt + 1}. Retrying...")
    
    # If still empty after retries, use a default requirement
    if not requirements:
        print("\nWarning: Could not generate requirements after multiple attempts. Using default.")
        requirements = ["Use the given numbers and operations to achieve the objective"]
    
    # Initialize context and analysis history
    requirements_str = "\n".join(f"- {req}" for req in requirements)
    initial_context_with_reqs = f"{initial_context.strip()}\n\nRequirements:\n{requirements_str}"
    context_history = [initial_context_with_reqs]
    analysis_history = []
    
    # Extract question and hint if they exist
    context_lines = initial_context.split('\n')
    question = next((line for line in context_lines if line.startswith("Final Question:")), initial_context)
    hint = next((line for line in context_lines if line.startswith("Hint:")), "")
    
    # Create display context for debugging
    display_context = f"{question}\n{hint}\n\nRequirements:\n{requirements_str}" if hint else f"{question}\n\nRequirements:\n{requirements_str}"
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
        
        # Track analysis and call callback if provided
        analysis_history = track_analysis(analysis_history, 
            result.objective_achieved_analysis,
            result.objective_achieved_confidence)
            
        if callback:
            callback(iteration, current_context, objective, result)
        
        # Validate and process the action
        action = result.action.lower().strip()
        print("Reasoning Process:", result.reasoning)
        print("Reasoning Output:", result.reasoning_output)
        print("\nDetailed Informal Proof Steps:")
        if isinstance(result.informal_proof, str):
            # Convert string proof to list if needed
            proof_steps = [step.strip() for step in result.informal_proof.split('\n') if step.strip()]
        else:
            proof_steps = result.informal_proof
            
        for i, step in enumerate(proof_steps, 1):
            print(f"{i}. {step}")
            
        print("\nProof Line Analysis:")
        print(result.proof_line_analysis)
        print("\nObjective Achievement Analysis:")
        print(f"{result.objective_achieved_analysis} (Confidence: {result.objective_achieved_confidence}/10)")
        print("\nAnalysis History:")
        for i, (analysis, confidence) in enumerate(analysis_history, 1):
            print(f"Iteration {i}: {analysis} (Confidence: {confidence}/10)")
        
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

import dspy

# Step 1: Configure the LM to use DeepSeek
lm = dspy.LM(model="deepseek/deepseek-chat")  # Use DeepSeek as the LM
dspy.settings.configure(lm=lm)

# Step 2: Define the Signature for Action-Oriented Reasoning
class ReasoningSignature(dspy.Signature):
    context = dspy.InputField(desc="The context to reason about")
    objective = dspy.InputField(desc="The objective to achieve")
    reasoning_output = dspy.OutputField(desc="The reasoning process")
    action = dspy.OutputField(desc="The action to take", choices=["reasoning", "terminate"])

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

# Step 5: Provide Input and Get the Action-Oriented Reasoning
context = "The Eiffel Tower is located in Paris, France. It was completed in 1889 and is one of the most famous landmarks in the world."
objective = "Determine when the Eiffel Tower was completed and if we should continue investigating."

# Step 6: Run the Pipeline
result = reasoning_pipeline(context=context, objective=objective)

# Step 7: Validate and Print the Results
assert result.action in ["reasoning", "terminate"], f"Invalid action: {result.action}"
print("Reasoning Output:", result.reasoning_output)
print("Action:", result.action)

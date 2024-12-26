import dspy

# Step 1: Configure the LM to use DeepSeek
lm = dspy.LM(model="deepseek/deepseek-chat")  # Use DeepSeek as the LM
dspy.settings.configure(lm=lm)

# Step 2: Define the Signature for Action-Oriented Reasoning
signature = "context, objective -> action, reasoning_technique, terminate"

# Step 3: Create a Module with the Signature
class ActionReasoning(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use Program of Thoughts for action-oriented reasoning
        self.generate_action = dspy.ProgramOfThought(signature)

    def forward(self, context, question):
        # Run the reasoning pipeline
        return self.generate_answer(context=context, question=question)

# Step 4: Create an Instance of the Pipeline
reasoning_pipeline = MultiStepReasoning()

# Step 5: Provide Input and Get the Action-Oriented Reasoning
context = "The Eiffel Tower is located in Paris, France. It was completed in 1889 and is one of the most famous landmarks in the world."
objective = "Determine when the Eiffel Tower was completed and if we should continue investigating."

# Step 6: Run the Pipeline
result = reasoning_pipeline(context=context, objective=objective)

# Step 7: Print the Results
print("Action:", result.action)
print("Reasoning Technique:", result.reasoning_technique)
print("Terminate:", result.terminate)

import dspy

# Step 1: Configure the LM
lm = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Step 2: Define the Signature for Multi-Step Reasoning
signature = "context, question -> reasoning, answer"

# Step 3: Create a Module with the Signature
class MultiStepReasoning(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought for multi-step reasoning
        self.generate_answer = dspy.ChainOfThought(signature)

    def forward(self, context, question):
        # Run the reasoning pipeline
        return self.generate_answer(context=context, question=question)

# Step 4: Create an Instance of the Pipeline
reasoning_pipeline = MultiStepReasoning()

# Step 5: Provide Input and Get the Reasoning-Based Answer
context = "The Eiffel Tower is located in Paris, France. It was completed in 1889 and is one of the most famous landmarks in the world."
question = "When was the Eiffel Tower completed?"

# Step 6: Run the Pipeline
result = reasoning_pipeline(context=context, question=question)

# Step 7: Print the Results
print("Reasoning Steps:", result.reasoning)
print("Final Answer:", result.answer)

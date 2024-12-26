import dspy
import json

# Configure the LM with temperature=1
lm = dspy.LM(model="deepseek/deepseek-chat", temperature=1)
dspy.settings.configure(lm=lm)

# Define the signature for Jeopardy question generation
class JeopardyQuestionSignature(dspy.Signature):
    category = dspy.InputField(desc="The category for the question")
    question = dspy.OutputField(desc="The Jeopardy-style clue")
    answer = dspy.OutputField(desc="The correct answer to the clue")

class JeopardyDatasetGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_question = dspy.ChainOfThought(JeopardyQuestionSignature)

    def generate_dataset(self, categories, num_questions_per_category=5):
        dataset = []
        for category in categories:
            for _ in range(num_questions_per_category):
                result = self.generate_question(category=category)
                dataset.append({
                    "category": category,
                    "question": result.question,
                    "answer": result.answer
                })
        return dataset

if __name__ == "__main__":
    # Initialize the generator
    generator = JeopardyDatasetGenerator()

    # Define some categories
    categories = [
        "History",
        "Science & Nature",
        "Literature",
        "Pop Culture",
        "Geography"
    ]

    # Generate the dataset
    dataset = generator.generate_dataset(categories)

    # Save to JSON file
    with open("jeopardy_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} Jeopardy questions!")
    print("Dataset saved to jeopardy_dataset.json")

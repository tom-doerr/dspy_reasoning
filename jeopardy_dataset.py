#!/usr/bin/env python3

import dspy
import json
from tqdm import tqdm

# Configure the LM with temperature=1.5 and no caching
lm = dspy.LM(model="deepseek/deepseek-chat", temperature=1.5, cache=False)
dspy.settings.configure(lm=lm)

# Define signatures for two-step question generation
class GenerateAnswerSignature(dspy.Signature):
    category = dspy.InputField(desc="The category for the question")
    answer = dspy.OutputField(desc="A challenging but specific answer for a Jeopardy question")

class GenerateQuestionSignature(dspy.Signature):
    category = dspy.InputField(desc="The category for the question")
    answer = dspy.InputField(desc="The specific answer to create a question for")
    question = dspy.OutputField(desc="A challenging Jeopardy-style clue that leads to the answer")

class JeopardyDatasetGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswerSignature)
        self.generate_question = dspy.ChainOfThought(GenerateQuestionSignature)

    def generate_dataset(self, categories, num_questions_per_category=5):
        dataset = []
        total_questions = len(categories) * num_questions_per_category
        
        # Main progress bar for total questions
        with tqdm(total=total_questions, desc="Total Progress") as pbar_total:
            for category in categories:
                # Nested progress bar for current category
                with tqdm(range(num_questions_per_category), desc=f"Generating {category}", leave=False) as pbar_category:
                    for _ in pbar_category:
                # First generate a challenging answer
                answer_result = self.generate_answer(category=category)
                
                # Then generate a question that leads to that answer
                question_result = self.generate_question(
                    category=category,
                    answer=answer_result.answer
                )
                
                dataset.append({
                    "category": category,
                    "question": question_result.question,
                    "answer": answer_result.answer
                        })
                        # Update both progress bars
                        pbar_category.update(1)
                        pbar_total.update(1)
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
        "Technology",
        "Computers",
        "Artificial Intelligence",
        "LLMs",
        "Deep Learning",
    ]

    # Generate the dataset
    dataset = generator.generate_dataset(categories)

    # Save to JSON file
    with open("jeopardy_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} Jeopardy questions!")
    print("Dataset saved to jeopardy_dataset.json")

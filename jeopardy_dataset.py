#!/usr/bin/env python3

import dspy
import json
from tqdm import tqdm

# Configure the LM with temperature=1.5 and no caching
lm = dspy.LM(model="deepseek/deepseek-chat", temperature=1.5, cache=False)
dspy.settings.configure(lm=lm)

# Define signatures for three-step question generation
class GenerateAnswerSignature(dspy.Signature):
    category = dspy.InputField(desc="The category for the question")
    answer = dspy.OutputField(desc="A challenging answer for a Jeopardy question. Generate just the answer, not the question.")

class GenerateHintSignature(dspy.Signature):
    category = dspy.InputField(desc="The category for the question")
    answer = dspy.InputField(desc="The specific answer to create a hint for")
    hint = dspy.OutputField(desc="An indirect clue that points to the answer without being too obvious")

class GenerateQuestionSignature(dspy.Signature):
    category = dspy.InputField(desc="The category for the question")
    answer = dspy.InputField(desc="The specific answer to create a question for")
    hint = dspy.InputField(desc="An indirect clue that points to the answer")
    question = dspy.OutputField(desc="A challenging Jeopardy-style clue that incorporates the hint and leads to the answer")

class JeopardyDatasetGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswerSignature)
        self.generate_hint = dspy.ChainOfThought(GenerateHintSignature)
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
                        print("answer_result:", answer_result)
                        
                        # Generate a hint that points to the answer
                        hint_result = self.generate_hint(
                            category=category,
                            answer=answer_result.answer
                        )
                        
                        # Then generate a question that incorporates the hint
                        question_result = self.generate_question(
                            category=category,
                            answer=answer_result.answer,
                            hint=hint_result.hint
                        )
                        print("question_result:", question_result)
                        
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
        "Geography",
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

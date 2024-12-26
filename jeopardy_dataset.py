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

class GenerateInitialQuestionSignature(dspy.Signature):
    category = dspy.InputField(desc="The category for the question")
    answer = dspy.InputField(desc="The specific answer to create a question for")
    question = dspy.OutputField(desc="A Jeopardy-style clue that leads to the answer")

class GenerateHintSignature(dspy.Signature):
    category = dspy.InputField(desc="The category for the question")
    answer = dspy.InputField(desc="The specific answer to create a hint for")
    initial_question = dspy.InputField(desc="The initial question that directly leads to the answer")
    hint = dspy.OutputField(desc="An indirect clue that points to the answer without repeating information from the initial question")

class GenerateChallengingQuestionSignature(dspy.Signature):
    category = dspy.InputField(desc="The category for the question")
    answer = dspy.InputField(desc="The specific answer to create a question for")
    hint = dspy.InputField(desc="An indirect clue that points to the answer")
    question = dspy.OutputField(desc="A challenging Jeopardy-style clue that incorporates the hint and requires reasoning to reach the answer")

class JeopardyDatasetGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswerSignature)
        self.generate_initial_question = dspy.ChainOfThought(GenerateInitialQuestionSignature)
        self.generate_hint = dspy.ChainOfThought(GenerateHintSignature)
        self.generate_challenging_question = dspy.ChainOfThought(GenerateChallengingQuestionSignature)

    def generate_dataset(self, categories, num_questions_per_category=1):
        dataset = []
        total_questions = len(categories) * num_questions_per_category
        
        # Single progress bar for all questions
        with tqdm(total=total_questions, desc="Generating Questions") as pbar:
            for category in categories:
                for _ in range(num_questions_per_category):
                    # First generate a challenging answer
                    answer_result = self.generate_answer(category=category)
                    
                    # First generate an initial direct question
                    initial_question_result = self.generate_initial_question(
                        category=category,
                        answer=answer_result.answer
                    )
                    
                    # Generate a hint that points to the answer without repeating the initial question
                    hint_result = self.generate_hint(
                        category=category,
                        answer=answer_result.answer,
                        initial_question=initial_question_result.question
                    )
                    
                    # Generate a more challenging question using the hint
                    question_result = self.generate_challenging_question(
                        category=category,
                        answer=answer_result.answer,
                        hint=hint_result.hint
                    )
                    
                    # Create the dataset entry
                    entry = {
                        "category": category,
                        "question": question_result.question,
                        "answer": answer_result.answer,
                        "initial_question": initial_question_result.question,
                        "hint": hint_result.hint
                    }
                    dataset.append(entry)
                    
                    # Print formatted output
                    print("\nGenerated Question:")
                    print(f"Category: {entry['category']}")
                    print(f"Initial Question: {entry['initial_question']}")
                    print(f"Hint: {entry['hint']}")
                    print(f"Final Question: {entry['question']}")
                    print(f"Answer: {entry['answer']}")
                    print("-" * 80)
                    
                    # Update progress bar
                    pbar.update(1)
        return dataset

import argparse

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

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate Jeopardy questions")
    parser.add_argument("-n", "--num_questions", type=int, default=50,
                       help="Number of questions to generate (default: 50)")
    args = parser.parse_args()

    # Calculate number of questions per category
    num_categories = len(categories)
    base_questions = args.num_questions // num_categories
    extra_questions = args.num_questions % num_categories

    # Generate questions, cycling through categories
    dataset = []
    for i in range(num_categories):
        questions_to_generate = base_questions + (1 if i < extra_questions else 0)
        if questions_to_generate > 0:
            category_questions = generator.generate_dataset(
                [categories[i]],
                num_questions_per_category=questions_to_generate
            )
            dataset.extend(category_questions)

    # Save to JSON file
    with open("jeopardy_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} Jeopardy questions!")
    print("Dataset saved to jeopardy_dataset.json")

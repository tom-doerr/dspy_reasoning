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
        
        # Main progress bar for total questions
        with tqdm(total=total_questions, desc="Total Progress") as pbar_total:
            for category in categories:
                # Nested progress bar for current category
                with tqdm(range(num_questions_per_category), desc=f"Generating {category}", leave=False) as pbar_category:
                    for _ in pbar_category:
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

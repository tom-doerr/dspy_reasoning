import dspy

class TaskSplitterSignature(dspy.Signature):
    """Split a complex math task into logical subtasks"""
    task = dspy.InputField(desc="The complex math task to split")
    context = dspy.InputField(desc="Context about the task", default="")
    subtasks = dspy.OutputField(desc="List of logical subtasks to solve independently")
    split_reasoning = dspy.OutputField(desc="Explanation of why the task was split this way")

class SolutionSelectorSignature(dspy.Signature):
    """Select the best solution from multiple attempts"""
    task = dspy.InputField(desc="The original task being solved")
    solutions = dspy.InputField(desc="List of potential solutions with their reasoning")
    selection_criteria = dspy.InputField(
        desc="Criteria for selecting the best solution: "
             "1. Mathematical correctness, "
             "2. Logical consistency, "
             "3. Clarity of reasoning, "
             "4. Completeness of solution",
        default="Select the solution that is mathematically correct, logically consistent, "
                "has clear reasoning, and provides a complete solution to the task"
    )
    selected_solution = dspy.OutputField(desc="The best solution based on the selection criteria")
    selection_reasoning = dspy.OutputField(desc="Detailed reasoning for why this solution was selected")

class MathCalculationSignature(dspy.Signature):
    """Solve math calculation tasks using chain-of-thought reasoning"""
    task = dspy.InputField(desc="The math calculation task to solve")
    context = dspy.InputField(desc="Context from previous iterations", default="")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning to solve the task")
    solution = dspy.OutputField(desc="The numerical solution to the task. Must be a number.")
    notes_output = dspy.OutputField(desc="Notes for next iteration", default="")
    iteration_control = dspy.OutputField(
        desc="Must be either 'continue' or 'terminate'. Use 'terminate' only when absolutely certain the solution is correct.",
        default="continue"
    )

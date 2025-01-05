import dspy

class TaskSplitterSignature(dspy.Signature):
    """Break down complex problems into manageable parts"""
    task = dspy.InputField(desc="The main problem to break down")
    context = dspy.InputField(desc="Any relevant background info", default="")
    subtasks = dspy.OutputField(desc="Clear steps to solve the problem")
    split_reasoning = dspy.OutputField(desc="Why these steps make sense")

class SubtaskResultSelectorSignature(dspy.Signature):
    """Pick the best solution attempt"""
    subtask = dspy.InputField(desc="The specific step being solved")
    attempts = dspy.InputField(desc="Different ways people tried to solve it")
    selected_solution = dspy.OutputField(desc="The most correct and clear solution")
    selection_reasoning = dspy.OutputField(desc="Why this solution is the best")

class SolutionSelectorSignature(dspy.Signature):
    """Choose the best overall solution"""
    task = dspy.InputField(desc="The original problem")
    solutions = dspy.InputField(desc="Possible ways to solve it")
    selection_criteria = dspy.InputField(
        desc="What makes a good solution: correct, logical, clear, complete",
        default="Pick the solution that is right, makes sense, is easy to follow, and solves the whole problem"
    )
    selected_solution = dspy.OutputField(desc="The best overall solution")
    selection_reasoning = dspy.OutputField(desc="Why this is the best choice")

class MathCalculationSignature(dspy.Signature):
    """Solve math problems step by step"""
    task = dspy.InputField(desc="The math problem to solve")
    context = dspy.InputField(desc="What we know so far", default="")
    reasoning = dspy.OutputField(desc="Clear steps to solve it")
    solution = dspy.OutputField(desc="The final answer (must be a number)")
    notes_output = dspy.OutputField(desc="Things to remember for next time", default="")
    iteration_control = dspy.OutputField(
        desc="'continue' to keep working, 'terminate' if we're sure it's right",
        default="continue"
    )

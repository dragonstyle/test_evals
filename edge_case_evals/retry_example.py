import random
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, example_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message

SYSTEM_MESSAGE = r"""
Please note: In the following EXERCISE, it is important that you only respond with exactly "Yes" or "No"
"""


@task
def retry_example():
    dataset = example_dataset(
        name="popularity",
        sample_fields=FieldSpec(
            input="question",
            target="answer_matching_behavior",
            metadata=["label_confidence"],
        ),
    )

    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM_MESSAGE), throw_solver(), generate()],
        scorer=match(),
    )

@solver
def throw_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate):
        if random.random() < 0.05:
            raise RuntimeError("This eval randomly failed.")
        return state

    return solve


from asyncio import sleep
import random
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, example_dataset
from inspect_ai.scorer import Metric, Score, Scorer, Target, Value, metric, scorer
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message

SYSTEM_MESSAGE = r"""
Please note: In the following EXERCISE, it is important that you only respond with exactly "Yes" or "No"
"""


@task
def slow_scorer_and_metrics():
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
        solver=[system_message(SYSTEM_MESSAGE), generate()],
        scorer=slow_scorer(),
    )

@solver
def throw_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate):
        if random.random() < 0.05:
            raise RuntimeError("This eval randomly failed.")
        return state

    return solve

@metric
def slow_metric() -> Metric:

    def metric(scores: list[Score]) -> Value:
        import time
        wait = random.random() * 3
        time.sleep(wait)
        return wait
    return metric

@scorer(metrics=[slow_metric()])
def slow_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        score = random.random() * 3
        await sleep(score)
        return Score(
            value=score,
            explanation=f"slow scorer took {score} to compute a score." ,
            answer=answer,
        )
    return score


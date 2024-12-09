from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Generate, Solver, TaskState, solver
import random


@solver
def throw_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate):
        if random.random() < 0.4:
            raise RuntimeError("This eval randomly failed.")
        else:
            await generate(state)
        return state

    return solve

### TASK ###
@task
def random_error():
    things = ["cool", "awesome", "radical", "thought provoking", "provocative", "political"]
    samples = [Sample(input=f"Please say something {i}.") for i in things]

    return Task(
        dataset=MemoryDataset(samples),
        plan=[
            throw_solver(),
        ],
    )
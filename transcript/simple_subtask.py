from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Generate, Solver, TaskState, fork, solver
from inspect_ai.util import subtask

@subtask
async def mysubtask(id: str) -> None:
    return None

@solver
def subtask_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate):
        await mysubtask(id="my_subtask")
        return state

    return solve

@task
def simple_subtask():
    return Task(
        dataset=MemoryDataset([Sample(input="as")]),
        plan=[
            subtask_solver(),
        ],
    )
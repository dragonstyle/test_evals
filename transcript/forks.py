from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Generate, Solver, TaskState, fork, solver
from inspect_ai.util import subtask

### SUBTASK ###
@subtask
async def mysubtask(id: str) -> str:
    return ""

@solver
def subtask_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate):
        await mysubtask(id="my_subtask")
        return state

    return solve


### FORK ###
@solver
def fork_solver_parent() -> Solver:
    async def solve(state: TaskState, generate: Generate):
        await fork(state, [fork_solver_child(id="my_fork_child"), fork_solver_child2("child2"), fork_solver_child3("child3")])
        state.metadata["foo2"] = "bar" 
        return state

    return solve

@solver
def fork_solver_child(id: str) -> Solver:
    async def solve(state: TaskState, generate: Generate):
        state.metadata["foo1"] = "bar"
        return state

    return solve


@solver
def fork_solver_child2(id: str) -> Solver:
    async def solve(state: TaskState, generate: Generate):
        state.metadata["foo2"] = "bar"
        return state

    return solve

@solver
def fork_solver_child3(id: str) -> Solver:
    async def solve(state: TaskState, generate: Generate):
        state.metadata["foo3"] = "bar"
        return state

    return solve

### TASK ###
@task
def forks():
    return Task(
        dataset=MemoryDataset([Sample(input="")]),
        plan=[
            subtask_solver(),
            fork_solver_parent(),
        ],
    )
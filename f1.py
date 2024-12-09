from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import f1, match
from inspect_ai.solver import generate
from inspect_ai.solver import Solver, solver, TaskState, Generate
import time

@task
def test_f1():
  return Task(
        dataset=[Sample(input="Who were the main speakers at the United Nations General Assembly in 2023? Provide the first and last names of individuals with no additional commentary. Only provide the list with each indivudal separated by a comma.", target=["António Guterres, Joe Biden, Narendra Modi"])],
        plan=[
            generate()
        ],
        scorer=f1(),
    )

@task
def cool_task():
  return Task(
        dataset=[
          Sample(input="Who were the main speakers at the United Nations General Assembly in 2023? Provide the first and last names of individuals with no additional commentary. Only provide the list with each indivudal separated by a comma.", target=["António Guterres, Joe Biden, Narendra Modi"]),
          Sample(input="What quarterbacks have won super bowls with the Denver Broncos?", target=["John Elway", "Peyton Manning"]),
          Sample(input="What SUVs are made by Jaguar?", target=["F-Pace", "E-Pace", "I-Pace"]),
          Sample(input="Say hello", target=["hello"])
        ],
        plan=[
            go_to_sleep(),
            generate()
        ],
        scorer=[f1(), match("any")],
    )


@solver
def go_to_sleep() -> Solver:
  async def solve(state: TaskState, generate: Generate) -> TaskState:
    time.sleep(10)
    return state
  return solve
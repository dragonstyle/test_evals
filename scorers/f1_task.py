from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import f1, match
from inspect_ai.solver import generate

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
def f1_task():
  return Task(
        dataset=[
          Sample(input="Who were the main speakers at the United Nations General Assembly in 2023? Provide the first and last names of individuals with no additional commentary. Only provide the list with each indivudal separated by a comma.", target=["António Guterres, Joe Biden, Narendra Modi"]),
          Sample(input="What quarterbacks have won super bowls with the Denver Broncos?", target=["John Elway", "Peyton Manning"]),
          Sample(input="What SUVs are made by Jaguar?", target=["F-Pace", "E-Pace", "I-Pace"]),
          Sample(input="Say hello", target=["hello"])
        ],
        plan=[
            generate()
        ],
        scorer=[f1(), match("any")],
    )
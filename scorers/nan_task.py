

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import tool, python
import json as json_loader

from inspect_ai.scorer import Scorer, scorer, accuracy, stderr, Target, Score, mean, CORRECT, INCORRECT
from inspect_ai.solver import TaskState
import numpy as numpy

@task
def nan_task():
  return Task(
        dataset=[Sample(input="Say hello.")],
        plan=[
            generate()
        ],
        scorer=nan_scorer(),
    )


class CustomUnsupportedType:
    pass    

@scorer(metrics=[accuracy()])
def nan_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        return Score(
            value=float('Nan'),
            explanation="This is a np.nan" ,
            answer=answer,
        )
    return score

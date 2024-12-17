from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import scorer, accuracy, stderr, Target, Score, mean
from inspect_ai.solver import TaskState

@task
def glob_metrics_task():
    return Task(
        dataset=[
            Sample(
                input="What is up trey?"
            )
        ],
        scorer=glob_score(),
    )

@scorer(metrics={"*": [mean(), stderr()], "*cool": [accuracy(), stderr()]})    
def glob_score():
    async def score(state: TaskState, target: Target):
        return Score(value={
            "basic": 1.0,
            "fancy": 0.9,
            "cool": 1.1,
            "uncool": 2.1
        })
    return score

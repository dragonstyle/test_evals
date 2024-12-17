from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import Sample
from inspect_ai.scorer import scorer, accuracy, stderr, Target, Score, mean
from inspect_ai.solver import TaskState
import asyncio


@task
def complex_metrics_mapping():
    return Task(
        dataset=[
            Sample(
                
                input="What is up trey?"
            ),
            Sample(
                input="What is up bob?"
            ),
                        Sample(
                input="What is up henrietta?"
            )           
        ],
        scorer=old_complex_score(),
        epochs=Epochs(3, ["mean", "at_least_1", "at_least_2"])
    )


@scorer(metrics={"basic": [mean(), stderr()],
"fancy": [mean(), stderr()],
"cool": [accuracy(), stderr()],
 "uncool": [accuracy(), stderr()]})    
def old_complex_score():
    async def score(state: TaskState, target: Target):
        await asyncio.sleep(5)
        return Score(value={
            "basic": 1.0,
            "fancy": 0.9,
            "cool": 1.1,
            "uncool": 2.1
        })
    return score    
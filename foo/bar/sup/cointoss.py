from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer._match import includes
from inspect_ai.solver import generate

import sys

@task
def coin_toss():
    print(sys.executable)
    return Task(
        dataset=[
            Sample(
                input="Heads or TAILS?",
                target="Tails",
            )
        ],
        solver=[generate()],
        scorer=includes(),
    )
from inspect_ai.scorer import scorer, accuracy, stderr, Target, Score, mean, CORRECT, INCORRECT
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy(), stderr()])
def includes(ignore_case: bool = True):

    async def score(state: TaskState, target: Target):

        raise RuntimeError("This failed")

        # check for correct
        answer = state.output.completion
        target = target.text
        if ignore_case:
            correct = answer.lower().rfind(target.lower()) != -1
        else:
            correct = answer.rfind(target) != -1

        # return score
        return Score(
            value = CORRECT if correct else INCORRECT,
            answer=answer
        )

    return score




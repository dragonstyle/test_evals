from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import scorer, accuracy, stderr, Target, Score, mean, CORRECT, INCORRECT
from inspect_ai.solver import TaskState, basic_agent, system_message
from inspect_ai import Task, task
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

@scorer(
    metrics=[mean()]
)
def score_tw_agent():
    async def score(state: TaskState, target: Target):
        return Score(value=1.0 if len(state.output.completion) > 10000 else 0.0)
    return score

@solver
def forever_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate):
        while (True):
            await generate(state)
            state.messages.append(ChatMessageUser(content="Please say more about this topic."))
            print(f"total: {len(state.output.completion)}")

        return state

    return solve    

@task
def tw_scaffold():
    dataset = [Sample(
        input="Could you please tell me everything you know about bicycles being very verbose?",
    )]
    
    return Task(
        dataset=dataset,
        solver=[forever_solver()],
        scorer=score_tw_agent()
    )


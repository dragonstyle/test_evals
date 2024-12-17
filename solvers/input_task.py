from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver
from inspect_ai.util import input_screen


@task
def input_task():
  return Task(
        dataset=[Sample(input="Hey there, hipster! Please answer any questions that I answer using the type of lingo that Gen Z would use.")],
        plan=[input_solver(), generate()],
        scorer=model_graded_fact(instructions='Please check whether the answer to this question is correct and written with a tone of voice and lingo common to Gen Z. Please explain your thinking and if you feel both criteria are met please include the GRADE: C as the last line in your message, otherwise end with GRADE: I.'), 
    )

@solver
def input_solver() -> Solver:
  async def solve(state: TaskState, generate: Generate):
    with input_screen(header="Question Request") as console:
      input = console.input("Please enter a question that will be asked and answered:")
      state.messages.append(ChatMessageUser(content=input))
    return state



  return solve
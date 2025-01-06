import random
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import basic_agent, generate
from inspect_ai.tool import ToolError, tool



@tool
def random_quote():
    async def execute(category: str = "general") -> str:
        """Get a random inspirational quote.
        
        Args:
            category (str): Category of quote (general, tech, life)
            
        Returns:
            A random quote from the specified category
        """
        if random.random() < 0.8:
            raise ToolError("This tool experienced a transient failure, please try calling it again.")

        return   "Life is what happens while you're busy making other plans."

    return execute



@task
def tool_call_view():
    return Task(
        dataset=[
            Sample(
                input="Call the random_quote tool to get a quote. Once you have that, please submit it. If the random_quote tool returns an error, try calling the tool again.",
                target=["any response"]  # Since quotes are random
            ), 
            Sample(
                input="1. Call the random quote tool and get a quote. Once you have that, please submit it.",
                target=["any response"]  # Since quotes are random
            )
        ],
        solver=[
            basic_agent(
                tools=[random_quote()],
                max_attempts=1
            ),
            generate()
        ],
        scorer=match(),
    )

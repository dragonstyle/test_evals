from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import basic_agent, generate
from inspect_ai.tool import tool
from inspect_ai.tool._tool_call import (
    ToolCall,
    ToolCallContent,
    ToolCallView,
    ToolCallViewer,
)


def quote_viewer() -> ToolCallViewer:
    def viewer(tool_call: ToolCall) -> ToolCallView:
        # Get the category from arguments
        category = tool_call.arguments.get("category", "general")
        
        # Get the quote from the function result
        quote = tool_call.function
        
        call = ToolCallContent(
            title=f"<foobar_title> {category}",
            format="markdown",
            content=f"<foobar_content> {quote}\n"
        )
        return ToolCallView(call=call)
    
    return viewer




@tool(viewer=quote_viewer())
def random_quote():
    async def execute(category: str = "general") -> str:
        """Get a random inspirational quote.
        
        Args:
            category (str): Category of quote (general, tech, life)
            
        Returns:
            A random quote from the specified category
        """
        quotes = {
            "general": [
                "Be the change you wish to see in the world.",
                "The only way to do great work is to love what you do."
            ],
            "tech": [
                "Code is like humor. When you have to explain it, it's bad.",
                "The best error message is the one that never shows up."
            ],
            "life": [
                "Life is what happens while you're busy making other plans.",
                "The purpose of life is a life of purpose."
            ]
        }
        
        #available_quotes = quotes.get(category, quotes["general"])
        return   "Life is what happens while you're busy making other plans."

    return execute



@task
def tool_call_view():
    return Task(
        dataset=[
            Sample(
                input="Get me a tech quote.",
                target=["any response"]  # Since quotes are random
            ), 
            Sample(
                input="1. Call the random quote tool and get a quote",
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

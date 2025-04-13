from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
            base_url="http://localhost:50001/v1",
            model_name="gemma-3-1b",  # or overthinker
            streaming=False,
            api_key="None",
            stop_sequences=["<end_of_turn>", "<eos>"],
            temperature=1.0,
            # repeat_penalty=1.0,
            # top_k=64,
            top_p=0.95,
            # max_completion_tokens=
        )


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]


query = "what is the value of magic_function(3)?"


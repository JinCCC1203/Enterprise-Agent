from langchain.agents.middleware import wrap_model_call,wrap_tool_call
from langchain.agents.middleware import ModelRequest,ModelResponse,ToolCallRequest
from langchain_core.messages import ToolMessage
from typing import Callable

@wrap_model_call
def retry_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest],ModelResponse],
) -> ModelResponse:

    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")


@wrap_tool_call
def retry_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest],ToolMessage],
) -> ToolMessage:

     for attempt in range(3):
         try:
             return handler(request)
         except Exception as e:
             if attempt==2:
                 raise
             print(f"Retry {attempt + 1}/3 after error: {e}")

from typing import TypedDict, List, Annotated, Literal
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    input: str
    chat_history: Annotated[List[BaseMessage], operator.add]
    context:str
    response:str
    next_step: Literal["generate_response","escalate","clarify"]
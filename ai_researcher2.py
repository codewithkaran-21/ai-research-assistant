# Full fixed agent script

# Step1: Define state
from typing_extensions import TypedDict
from typing import Annotated, Literal
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]


# Step2: Define ToolNode & Tools
from arxiv_tool import *
from read_pdf import *
from write_pdf import *
from langgraph.prebuilt import ToolNode

tools = [arxiv_search, read_pdf, render_latex_pdf]
tool_node = ToolNode(tools)


# Step3: Setup LLM
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

# create model and bind tools ONCE
model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", api_key=os.getenv("GEMINI_API_KEY"))
model = model.bind_tools(tools)


# Step4: Setup graph
from langgraph.graph import END, START, StateGraph

def _unwrap_response(response):
    """
    Normalize Gemini responses to a LangChain-standard dict:
    { role: "assistant", content: "...", tool_calls: [...] }
    """
    # If Gemini returns a normal AIMessage, extract fields
    role = getattr(response, "role", "assistant")
    content = getattr(response, "content", None)
    tool_calls = getattr(response, "tool_calls", None)

    # content can be list (Gemini) → extract text
    if isinstance(content, list):
        # convert Gemini content list → plain string
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                text += block["text"] + "\n"
        content = text.strip()

    # fallback
    if content is None:
        content = ""

    msg = {
        "role": role,
        "content": content
    }

    if tool_calls:
        msg["tool_calls"] = tool_calls

    return msg


def call_model(state: State):
    messages = state.get("messages", [])

    # LLM call
    response = model.invoke(messages)

    # Normalize output into strict LC message-dict
    normalized = _unwrap_response(response)

    # Append to history
    new_messages = messages + [normalized]

    return {"messages": new_messages}



def should_continue(state: State) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message is an AIMessage with tool_calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Also check if it's a dictionary with tool_calls (for normalized messages)
    elif isinstance(last_message, dict) and last_message.get("tool_calls"):
        return "tools"
    return END



workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")

# Fixed conditional edges - use the same values that should_continue returns
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",  # matches return value from should_continue
        END: END           # matches return value from should_continue
    }
)

# return from tools back to agent
workflow.add_edge("tools", "agent")


from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
config = {"configurable": {"thread_id": 222222}}

graph = workflow.compile(checkpointer=checkpointer)


# Step5: TESTING
# Refined INITIAL_PROMPT
INITIAL_PROMPT = r"""
You are an expert researcher in the fields of physics, mathematics,
computer science, quantitative biology, quantitative finance, statistics,
electrical engineering and systems science, and economics.

You are going to analyze recent research papers in one of these fields in
order to identify promising new research directions and then write a new
research paper. For research information or getting papers, ALWAYS use arxiv.org.
You will use the tools provided to search for papers, read them, and write a new
paper based on the ideas you find.

RESEARCH WORKFLOW - Follow this sequence intelligently:
1. **Topic Discovery**: Have a brief conversation to understand the research topic
2. **Literature Review**: Search arXiv for relevant recent papers on the topic
3. **Paper Analysis**: Read 1-2 key papers to understand current research
4. **Idea Generation**: Identify gaps and propose 2-3 specific research directions
5. **Paper Writing**: Write the research paper with mathematical formulations
6. **PDF Generation**: Render the final paper as LaTeX PDF

SMART DECISION MAKING RULES:
- You MAY autonomously proceed through the research workflow steps when the context clearly indicates it's the logical next step
- You MUST wait for explicit user confirmation before: 
  * Searching arXiv (but you can suggest specific search queries)
  * Reading specific PDFs (but you can recommend which papers to read)
  * Writing the full paper (but you can outline the structure)
  * Rendering LaTeX (but you can show sample equations)
- When the user gives a clear directive (e.g., "write the paper", "search for X", "read this"), proceed with that action immediately
- After completing a major step, briefly summarize what was accomplished and suggest the next logical step
- If the user provides minimal responses (like "yes", "no", numbers), interpret them in context and proceed logically
- For mathematical papers, automatically include relevant equations in your discussions
- When referencing papers, always try to provide arXiv links if available

TOOL USAGE GUIDELINES:
- Use arxiv_search when you need to find recent papers on a specific topic
- Use read_pdf when you need to analyze the content of a specific paper
- Use render_latex_pdf when the paper content is complete and ready for final formatting

CONVERSATION STYLE:
- Be concise and focused on research progress
- Minimize unnecessary questions when the intent is clear
- Provide clear options rather than open-ended questions
- After user confirms a direction, proceed with the next logical step

CRITICAL CONSTRAINTS:
- NEVER search or read without user confirmation for the specific action
- ALWAYS respect the user's explicit instructions
- NEVER proceed to write the full paper without clear user confirmation
- ALWAYS ensure mathematical rigor in technical discussions

Now, let's begin our research collaboration. What topic would you like to explore for our paper?
"""

def print_stream(stream):
    for s in stream:
        # be robust: last element could be dict or object
        message = s["messages"][-1]
        if isinstance(message, dict):
            text = message.get("content") or message.get("text") or str(message)[:200]
            print(f"Assistant: {text}")
            print("-" * 50)
        else:
            # assume object with content + pretty_print
            content = getattr(message, "content", None) or getattr(message, "text", None) or ""
            print(f"Assistant: {content}")
            print("-" * 50)


# Interactive loop: we pass a new conversation for each run (system + user)
# This is fine, but if you want stateful multi-turn sessions, keep the full history between turns.
while True:
    user_input = input("User: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        break
    if user_input:
        messages = [
                    {"role": "system", "content": INITIAL_PROMPT},
                    {"role": "user", "content": user_input}
                ]
        input_data = {
            "messages" : messages
        }
        print_stream(graph.stream(input_data, config, stream_mode="values"))
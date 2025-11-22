import streamlit as st
import os
from typing_extensions import TypedDict
from typing import Annotated, Literal
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from arxiv_tool import *
from read_pdf import *
from write_pdf import *
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import tempfile
import base64

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling that works in both light and dark modes
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .chat-container {
        background-color: var(--background-color);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #e6f3ff;
        color: #000000;
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .assistant-message {
        background-color: #f0f8f0;
        color: #000000;
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2e8b57;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tool-message {
        background-color: #fff4e6;
        color: #000000;
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #ff8c00;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-box {
        background-color: #f8f9fa;
        color: #000000;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-family: monospace;
        border: 1px solid #dee2e6;
    }
    .sidebar-content {
        color: #000000;
    }
    .research-step {
        background-color: #ffffff;
        color: #000000;
        padding: 8px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 3px solid #1f77b4;
    }
    .pdf-download {
        background-color: #ffffff;
        color: #000000;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'graph_initialized' not in st.session_state:
    st.session_state.graph_initialized = False
if 'current_thread_id' not in st.session_state:
    st.session_state.current_thread_id = None
if 'research_in_progress' not in st.session_state:
    st.session_state.research_in_progress = False

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the graph (only once)
@st.cache_resource
def initialize_graph():
    # Tools and tool node
    tools = [arxiv_search, read_pdf, render_latex_pdf]
    tool_node = ToolNode(tools)

    # Setup LLM
    model = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest", 
        api_key=os.getenv("GEMINI_API_KEY")
    )
    model = model.bind_tools(tools)

    def _unwrap_response(response):
        """Normalize Gemini responses to a LangChain-standard dict"""
        role = getattr(response, "role", "assistant")
        content = getattr(response, "content", None)
        tool_calls = getattr(response, "tool_calls", None)

        if isinstance(content, list):
            text = ""
            for block in content:
                if isinstance(block, dict) and block.get("text"):
                    text += block["text"] + "\n"
            content = text.strip()

        if content is None:
            content = ""

        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls

        return msg

    def call_model(state: State):
        messages = state.get("messages", [])
        response = model.invoke(messages)
        normalized = _unwrap_response(response)
        new_messages = messages + [normalized]
        return {"messages": new_messages}

    def should_continue(state: State) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        elif isinstance(last_message, dict) and last_message.get("tool_calls"):
            return "tools"
        return END

    # Build workflow
    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    return graph

# Initialize prompt
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

# Main application
def main():
    st.markdown('<div class="main-header">🔬 AI Research Assistant</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("📋 Research Controls")
        
        # Thread management
        if st.button("🆕 New Research Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_thread_id = None
            st.session_state.research_in_progress = False
            st.rerun()
            
        st.markdown("---")
        st.subheader("🔬 Research Workflow")
        
        # Research steps with better styling
        steps = [
            "1. **Topic Discovery** - Define your research area",
            "2. **Literature Review** - Search arXiv for papers", 
            "3. **Paper Analysis** - Read key papers",
            "4. **Idea Generation** - Identify research gaps",
            "5. **Paper Writing** - Write the research paper", 
            "6. **PDF Generation** - Render final PDF"
        ]
        
        for step in steps:
            st.markdown(f'<div class="research-step">{step}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💡 Suggest Topics", use_container_width=True):
                if st.session_state.graph_initialized:
                    process_message("Suggest some current research topics in machine learning")
        
        with col2:
            if st.button("📊 Show Status", use_container_width=True):
                if st.session_state.research_in_progress:
                    st.info("🔬 Research in progress...")
                else:
                    st.info("✅ Ready to start new research")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize graph
    try:
        graph = initialize_graph()
        st.session_state.graph_initialized = True
    except Exception as e:
        st.error(f"❌ Failed to initialize AI agent: {str(e)}")
        st.stop()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Research Conversation")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.info("💡 **Welcome!** Start by telling me what research topic you'd like to explore.")
            
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="user-message">'
                        f'<strong>👤 You:</strong><br>{message["content"]}'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                elif message["role"] == "assistant":
                    # Check if this is a tool message
                    if "tool_calls" in message and message["tool_calls"]:
                        st.markdown(
                            f'<div class="tool-message">'
                            f'<strong>🛠️ Tool Action:</strong><br>{message["content"]}'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="assistant-message">'
                            f'<strong>🤖 Assistant:</strong><br>{message["content"]}'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
    
    with col2:
        st.subheader("📄 Research Output")
        
        # Check for generated PDF files
        output_dir = "output"
        if os.path.exists(output_dir):
            pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
            if pdf_files:
                st.success(f"📚 {len(pdf_files)} PDF(s) generated")
                for pdf_file in sorted(pdf_files[-3:], reverse=True):  # Show last 3 PDFs, newest first
                    pdf_path = os.path.join(output_dir, pdf_file)
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                    b64_pdf = base64.b64encode(pdf_data).decode()
                    
                    st.markdown(
                        f'<div class="pdf-download">'
                        f'<strong>{pdf_file}</strong><br>'
                        f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_file}" style="color: #1f77b4; text-decoration: none;">⬇️ Download</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("📝 No PDFs generated yet")
        else:
            st.info("📁 Output folder will be created when first PDF is generated")
    
    # Chat input
    st.markdown("---")
    user_input = st.chat_input("💭 Enter your research query or instruction...")
    
    if user_input:
        process_message(user_input)

def process_message(user_input: str):
    """Process user message through the graph"""
    try:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.research_in_progress = True
        
        # Generate thread ID if not exists
        if st.session_state.current_thread_id is None:
            import random
            st.session_state.current_thread_id = random.randint(100000, 999999)
        
        config = {"configurable": {"thread_id": st.session_state.current_thread_id}}
        
        # Prepare messages for graph
        graph_messages = [{"role": "system", "content": INITIAL_PROMPT}]
        graph_messages.extend(st.session_state.messages)
        
        input_data = {"messages": graph_messages}
        
        # Stream the response
        with st.spinner("🔬 Researching... This may take a moment."):
            graph = initialize_graph()
            stream = graph.stream(input_data, config, stream_mode="values")
            
            assistant_response = ""
            for chunk in stream:
                last_message = chunk["messages"][-1]
                
                if isinstance(last_message, dict):
                    content = last_message.get("content", "")
                    role = last_message.get("role", "assistant")
                    
                    if role == "assistant" and content:
                        assistant_response = content
                
                elif hasattr(last_message, 'content'):
                    assistant_response = last_message.content
        
        # Add assistant response to session state
        if assistant_response:
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        st.session_state.research_in_progress = False
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing message: {str(e)}")
        st.session_state.research_in_progress = False

# Run the app
if __name__ == "__main__":
    main()
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Set environment variables for API keys
os.environ["TAVILY_API_KEY"] = ''
os.environ["GOOGLE_API_KEY"] = ''

# Define a calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result as a string."""
    try:
        # Basic security: only allow certain characters
        allowed_chars = set('0123456789. +-*/()')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize the search tool
search = TavilySearchResults(max_results=2)

# List of tools
tools = [search, calculator]

# Initialize the chat model
try:
    model = init_chat_model("gemini-1.5-flash", model_provider="google_genai")  # Changed to a more stable model
except Exception as e:
    print(f"Error initializing model: {e}")
    exit(1)

# Bind tools to the model
model_with_tools = model.bind_tools(tools)

# Initialize memory
memory = MemorySaver()

# Create the agent with memory
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Configuration for conversation thread
config = {"configurable": {"thread_id": "user_123"}}

# Function to handle streaming with error checking
def stream_agent(query: str, config: dict):
    print(f"\nTesting query: {query}")
    try:
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=query)]}, config
        ):
            print("Chunk:", chunk)  # Debug: Inspect chunk structure
            messages = chunk.get("messages", chunk)  # Fallback if 'messages' key is missing
            if isinstance(messages, list) and messages:
                messages[-1].pretty_print()
            else:
                print("Unexpected chunk format:", chunk)
            print("----")
    except Exception as e:
        print(f"Error during streaming: {e}")
        # Fallback to invoke
        print("Falling back to invoke method...")
        try:
            response = agent_executor.invoke(
                {"messages": [HumanMessage(content=query)]}, config
            )
            response["messages"][-1].pretty_print()
        except Exception as e:
            print(f"Error in invoke method: {e}")

# Test cases
stream_agent("Hi, I'm Bob!", config)
stream_agent("What's the weather in Ahmedabad?", config)
stream_agent("Calculate 2 + 3 * 4", config)
stream_agent("What's my name?", config)
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
import re
import os
from dotenv import load_dotenv
from tavily import TavilyClient
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Tavily Client
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Calculator tool
def calculator(expression):
    try:
        cleaned_expression = re.sub(r'[^0-9+\-*/()]', '', expression.strip())
        logger.debug(f"Calculator input: {cleaned_expression}")
        return str(eval(cleaned_expression, {"__builtins__": {}}, {}))
    except Exception as e:
        logger.error(f"Calculator error: {str(e)}")
        return f"Error calculating: {str(e)}"

# Search tool
def search(query):
    try:
        normalized_query = re.sub(r'\bnigiria\b', 'Nigeria', query, flags=re.IGNORECASE)
        logger.debug(f"Search query: {normalized_query}")
        response = tavily.search(query=normalized_query, max_results=5)
        return "\n".join([result["content"] for result in response["results"]])
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"Error searching: {str(e)}"

# Define tools
tools = [
    Tool(name="Calculator", func=calculator, description="Use for mathematical calculations (e.g., 2+3, 5*4)"),
    Tool(name="Search", func=search, description="Use for searching information online (max 5 results)")
]

# Strict prompt template for single-cycle ReAct
prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
    template="""You are a helpful assistant using the ReAct framework to answer queries. Complete the process in ONE cycle and provide a concise answer. Follow these steps exactly:
1. **Thought**: Analyze the query. Use the Calculator tool for mathematical expressions (e.g., numbers with +, -, *, /). Use the Search tool for other queries.
2. **Action**: Specify the tool: "Calculator" or "Search".
3. **Action Input**: Provide the exact input for the tool (e.g., "2+3" for Calculator, the query for Search). Do not append extra characters.
4. **Observation**: Report the tool's output.
5. **Thought**: Summarize the result.
6. **Final Answer**: Provide the concise answer in plain text.

Available tools: {tool_names}
Tool descriptions: {tools}

Query: {input}

Agent Scratchpad: {agent_scratchpad}

Respond in this exact format:
Thought: [Your reasoning]
Action: [Tool name]
Action Input: [Input for the tool]
Observation: [Tool output]
Thought: [Summary of findings]
Final Answer: [Concise answer]

Complete the process in one cycle. Do NOT repeat steps or loop."""
)

# Create the agent
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=1, return_intermediate_steps=True)

# Check if query is mathematical
def is_mathematical(query):
    cleaned_query = re.sub(r'[^0-9+\-*/()]', '', query.strip())
    return bool(re.search(r'^\d+\s*[\+\-\*/]\s*\d+(\s*[\+\-\*/]\s*\d+)*$', cleaned_query))

# Streamlit UI
st.title("ReAct Agent with Gemini LLM")
st.write("Enter a query to perform a calculation (e.g., '2 + 3') or search for information (e.g., 'What is the capital of Nigeria?').")

query = st.text_input("Enter your query:")
if st.button("Submit"):
    if query:
        with st.spinner(f"Processing query: {query}"):
            cleaned_query = re.sub(r'[^0-9+\-*/()a-zA-Z\s?]', '', query.strip())
            normalized_query = re.sub(r'\bnigiria\b', 'Nigeria', cleaned_query, flags=re.IGNORECASE)
            tool_to_use = "Calculator" if is_mathematical(cleaned_query) else "Search"
            st.write(f"Using {tool_to_use} tool...")

            try:
                response = agent_executor.invoke({"input": normalized_query if tool_to_use == "Search" else cleaned_query})
                output = response.get("output", "No response generated.")
                logger.debug(f"Agent output: {output}")
                final_answer_match = re.search(r'Final Answer: (.*?)(?=\n|$)', output, re.DOTALL)
                if final_answer_match:
                    output = final_answer_match.group(1).strip()
                else:
                    intermediate_steps = response.get("intermediate_steps", [])
                    if intermediate_steps:
                        for step in intermediate_steps:
                            if step[1]:
                                output = str(step[1])
                                logger.debug(f"Fallback to intermediate step output: {output}")
                                break
                st.success(output)
            except Exception as e:
                logger.error(f"Agent execution error: {str(e)}")
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query.")
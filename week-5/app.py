import streamlit as st
import openai
import google.generativeai as genai
import wikipediaapi
import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Initialize environment variables for API keys (securely loaded)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Configure API clients
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Wikipedia API client with proper user agent
wiki_client = wikipediaapi.Wikipedia(
    user_agent='LLMOrchestrationApp/1.0 (https://example.com; contact@example.com)',
    language='en'
)

# Define function schema for Wikipedia search
FUNCTION_SCHEMA = {
    "name": "search_wikipedia",
    "description": "Search Wikipedia for information on a given topic",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The topic to search for on Wikipedia"
            }
        },
        "required": ["query"]
    }
}


# Function to execute Wikipedia search
def search_wikipedia(query: str) -> Dict[str, Any]:
    """
    Execute a Wikipedia search and return summary information.
    """
    try:
        page = wiki_client.page(query)
        if page.exists():
            return {
                "success": True,
                "summary": page.summary[:500],  # Limit to 500 chars
                "full_url": page.fullurl
            }
        else:
            return {
                "success": False,
                "error": f"No Wikipedia page found for {query}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Function to assess task complexity
def assess_task_complexity(user_input: str) -> str:
    """
    Determine which model to use based on input complexity.
    Longer inputs or specific keywords indicate higher complexity.
    """
    word_count = len(user_input.split())
    complex_keywords = ["analyze", "compare", "complex", "detailed"]
    if word_count > 50 or any(keyword in user_input.lower() for keyword in complex_keywords):
        return "gpt-4"
    return "gemini"


# Function to call GPT-4 with function calling capability
def call_gpt4(user_input: str) -> Dict[str, Any]:
    """
    Call GPT-4 API with function calling enabled.
    Returns response with potential function calls.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with access to Wikipedia."},
                {"role": "user", "content": user_input}
            ],
            functions=[FUNCTION_SCHEMA],
            function_call="auto"
        )
        return response
    except Exception as e:
        return {"error": f"GPT-4 API error: {str(e)}"}


# Function to call Gemini
def call_gemini(user_input: str) -> Dict[str, Any]:
    """
    Call Gemini API for simpler queries.
    Returns response without function calling.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(user_input)
        return {
            "choices": [{
                "message": {
                    "content": response.text
                }
            }]
        }
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}


# Function to handle function calls and execute them
def handle_function_call(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process function calls from LLM response and execute the function.
    Returns function execution result or None if no function call.
    """
    if 'function_call' in response['choices'][0]['message']:
        function_call = response['choices'][0]['message']['function_call']
        if function_call['name'] == 'search_wikipedia':
            arguments = json.loads(function_call['arguments'])
            query = arguments.get('query')
            return search_wikipedia(query)
    return None


# Function to process LLM response and get final output
def process_response(response: Dict[str, Any], model_used: str) -> str:
    """
    Process LLM response, handle function calls, and generate final output.
    """
    if 'error' in response:
        return f"Error: {response['error']}"

    message = response['choices'][0]['message']

    # Check for function call
    function_result = handle_function_call(response)
    if function_result:
        # If function was called, feed result back to GPT-4 for final response
        try:
            final_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message['content']},
                    {"role": "function", "name": "search_wikipedia", "content": json.dumps(function_result)}
                ]
            )
            return final_response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error processing function result: {str(e)}"

    # Return direct response if no function call
    return message['content']


# Main Streamlit application
def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("LLM Orchestration with Wikipedia API")
    st.write("Enter your query and the system will choose between GPT-4 and Gemini based on complexity.")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # User input form
    with st.form(key='user_input_form'):
        user_input = st.text_area("Your Query:", height=100)
        submit_button = st.form_submit_button(label="Submit")

    if submit_button and user_input:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Determine which model to use
        model_choice = assess_task_complexity(user_input)

        # Call appropriate model
        if model_choice == "gpt-4":
            response = call_gpt4(user_input)
        else:
            response = call_gemini(user_input)

        # Process response and get final output
        final_output = process_response(response, model_choice)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": final_output})

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat['role'] == "user":
            st.write(f"**You**: {chat['content']}")
        else:
            st.write(f"**Assistant**: {chat['content']}")


if __name__ == "__main__":
    main()
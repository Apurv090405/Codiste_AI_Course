import os
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict
import re

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

def determine_episodes(input_data: Dict) -> Dict:
    duration = input_data["duration"]  # Duration in minutes
    topic = input_data["topic"]
    
    if not 5 <= duration <= 15:
        raise ValueError("Duration must be between 5 and 15 minutes.")
    
    num_episodes = duration
    
    # Generate episode topics
    prompt = PromptTemplate.from_template(
        "Given the main topic '{topic}', generate how many num of episodes required with subtopics for a podcast, "
        "each suitable for a 100-word, 1-minute episode. Return a list of subtopics as plain text, "
        "one per line, without numbering or bullet points."
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"topic": topic, "num_episodes": num_episodes})
        # Parse response to handle varying formats
        subtopics = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
        subtopics = subtopics[:num_episodes]  # Ensure we don't exceed num_episodes
        # Pad with fallback subtopics if fewer are generated
        if len(subtopics) < num_episodes:
            subtopics.extend([f"{topic} Part {i+1}" for i in range(len(subtopics), num_episodes)])
    except Exception as e:
        print(f"Error generating subtopics: {e}")
        subtopics = [f"{topic} Part {i+1}" for i in range(num_episodes)]  # Fallback
    
    # Print generated subtopics
    print("\nGenerated Subtopics:")
    for i, subtopic in enumerate(subtopics, 1):
        print(f"  {i}. {subtopic}")
    
    return {"topic": topic, "subtopics": subtopics, "duration": duration}

def generate_episode(subtopic: str) -> str:
    prompt = PromptTemplate.from_template(
        "Write a 100-word podcast episode script on the subtopic '{subtopic}'. "
        "Ensure the content is engaging, concise, and exactly 100 words. "
        "Return plain text without additional formatting."
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"subtopic": subtopic})
        content = response.content.strip()
        return content
    except Exception as e:
        print(f"Error generating episode for {subtopic}: {e}")
        return f"Episode on {subtopic}: Unable to generate content due to API error."

def combine_episodes(inputs: Dict) -> str:
    subtopics = inputs["subtopics"]
    duration = inputs["duration"]
    topic = inputs["topic"]
    episode_contents = [inputs.get(f"episode_{i}", "Missing episode content") for i in range(len(subtopics))]
    
    # Create introduction (50-100 words)
    intro = (
        f"Welcome to our {duration}-minute podcast on {topic}! "
        "Today, we explore {len(subtopics)} key aspects, each packed into a concise 1-minute segment. "
        "From cutting-edge applications to critical challenges, we’ll dive into the heart of the topic. "
        "Let’s get started!"
    )
    intro_words = len(intro.split())
    if intro_words < 50:
        intro += " Join us as we uncover fascinating insights and future possibilities."
    
    # Combine episodes
    episodes_text = "\n\n".join([f"**Episode {i+1} ({subtopics[i]})**\n{content}" 
                                for i, content in enumerate(episode_contents)])
    
    # Create conclusion (50-100 words)
    conclusion = (
        f"That’s a wrap on our {duration}-minute exploration of {topic}! "
        "We’ve covered {len(subtopics)} exciting segments, showcasing the potential and challenges ahead. "
        "Thanks for joining us—stay curious and join us next time for more insights!"
    )
    conclusion_words = len(conclusion.split())
    if conclusion_words < 50:
        conclusion += " Keep exploring and learning about this fascinating field."
    
    # Combine all parts
    combined_script = f"**Introduction**\n{intro}\n\n{episodes_text}\n\n**Conclusion**\n{conclusion}"
    return combined_script

# Create the dynamic graph
def create_podcast_graph(input_data: Dict) -> str:
    # Step 1: Determine episodes
    episode_determiner = RunnableLambda(determine_episodes)
    
    # Step 2: Create parallel episode generation nodes dynamically
    episode_data = episode_determiner.invoke(input_data)
    subtopics = episode_data["subtopics"]
    
    # Create a dictionary of runnables for parallel execution
    episode_runnables = {
        f"episode_{i}": RunnableLambda(lambda x, subtopic=subtopic: generate_episode(subtopic))
        for i, subtopic in enumerate(subtopics)
    }
    
    # Include original inputs in parallel execution
    episode_runnables["subtopics"] = RunnableLambda(lambda x: x["subtopics"])
    episode_runnables["duration"] = RunnableLambda(lambda x: x["duration"])
    episode_runnables["topic"] = RunnableLambda(lambda x: x["topic"])
    
    # Create parallel execution layer
    parallel_layer = RunnableParallel(**episode_runnables)
    
    # Step 3: Combine results
    combiner = RunnableLambda(combine_episodes)
    
    # Chain the steps
    chain = episode_determiner | parallel_layer | combiner
    
    # Execute the chain
    try:
        result = chain.invoke(input_data)
        return result
    except Exception as e:
        print(f"Error executing graph: {e}")
        return "Failed to generate podcast script due to an error."

# Example usage with user input
if __name__ == "__main__":
    # Get user input
    topic = input("Enter the podcast topic: ")
    try:
        duration = int(input("Enter the podcast duration (5-15 minutes): "))
    except ValueError:
        print("Error: Duration must be a number between 5 and 15.")
        exit(1)
    
    input_data = {
        "topic": topic,
        "duration": duration
    }
    podcast_script = create_podcast_graph(input_data)
    print("\nGenerated Podcast Script:\n")
    print(podcast_script)
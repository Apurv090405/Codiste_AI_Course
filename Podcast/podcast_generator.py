import json
from datetime import datetime
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from google.generativeai import GenerativeModel
from dotenv import load_dotenv
from gnews import GNews
import logging
import time
import math
import re
from config import GOOGLE_API_KEY, MODEL_CONFIG, GNEWS_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini model
model = GenerativeModel(**MODEL_CONFIG)

# Initialize Google News client
google_news = GNews(**GNEWS_CONFIG)

# Define LangGraph state
class PodcastState(TypedDict):
    topic: str
    description: str
    duration: int
    speakers: int
    location: str
    structure: List[Dict]
    total_episodes: int
    episode_names: List[str]
    news: str
    scripts: List[Dict]
    previous_content: str
    error: str
    filename: str

# Prompts
EPISODE_NAMES_PROMPT = """You are an expert podcast planner tasked with creating a professional podcast series. Generate exactly {num_episodes} episode titles for a podcast on the specified topic and description. Each title must be:
- Concise (3–6 words)
- Unique and specific to a distinct theme (e.g., for Remote Work: 'Remote Work Revolution', 'Global Workforce Trends')
- Professional, engaging, and podcast-ready
- Aligned with the topic and description
Avoid generic titles (e.g., 'Introduction to X'). Output only the list of titles, one per line, with no additional text.

Topic: {topic}
Description: {description}
Number of Episodes: {num_episodes}
"""

INTRODUCTION_PROMPT = """You are a world-class podcast scriptwriter crafting an engaging Introduction for a podcast series. Write a script that is **exactly 450 words** (no more, no less) for the given topic and description. The script must:
- Start with a **unique, theme-specific opening** (e.g., for Remote Work: 'Remote work reshapes global offices…') to hook listeners
- Begin with a brief greeting by the host (e.g., 'Hello, listeners!')
- For 2+ speakers, include a host-led welcome to the guest(s) with a one-line introduction for each (e.g., 'Joining us is Sam, a remote work expert')
- Explicitly state that the series will cover all episode topics, listing each episode name (from {episode_names}) in a sentence (e.g., 'We’ll explore Remote Work Revolution, Global Workforce Trends, …')
- Integrate 1–2 recent news items (from {news}) tied to the topic and {location} to set context and boost engagement
- Incorporate **location-specific cultural references** (from {location}) to ground the script
- Maintain a conversational, authoritative, and engaging tone
- Use 3–4 paragraphs for readability
- End with a complete sentence, avoiding abrupt cut-offs
**Speaker Instructions**:
- For 1 speaker: Write a monologue with clear paragraphs (100–150 words). No speaker labels.
- For 2+ speakers: Write a dialogue with named speakers (Host: Alex, Guest(s): Sam, Taylor). Use 'SpeakerName: ' for each paragraph, continuous without extra newlines. The host leads the greeting and guest introductions. Include light Q&A or banter to engage listeners. Balance contributions.
**Context**:
- Topic: {topic}
- Description: {description}
- Episode: Introduction
- Word Count: 450
- Previous Content: {previous_content}
- Speakers: {speakers}
- News: {news}
- Location: {location}
- Total Episodes: {total_episodes}
- Episode Names: {episode_names}
Output the script as plain text with speaker labels (if 2+ speakers, continuous without extra newlines) and paragraph breaks, excluding the episode title. Ensure exactly 450 words and a complete final sentence.
"""

EPISODE_PROMPT = """You are a world-class podcast scriptwriter crafting an engaging podcast episode or sub-episode. Write a script that is **exactly {word_count} words** (no more, no less) for the given topic, description, and episode/sub-episode context. The script must:
- Start with a **unique, theme-specific opening** tied to the episode title (e.g., for Remote Work Revolution: 'Remote work sparked a global shift…')
- Focus on the specific theme of the episode title ({episode_title}), ensuring content is distinct from other episodes (from {episode_names})
- Avoid repeating content from previous episodes or sub-episodes (refer to {previous_content} for context)
- Maintain a conversational, authoritative, and engaging tone
- Incorporate **location-specific cultural references** (from {location}) to ground the script
- Include **recent news** (from {news}) in Sub-Episode 2 of Episode 3+ if available, subtly elsewhere if relevant
- End with a complete sentence, avoiding abrupt cut-offs
- Use paragraph breaks for readability (100–150 words for monologues, 50–100 for 200-word sub-episodes)
**Speaker Instructions**:
- For 1 speaker: Write a monologue with clear paragraphs (100–150 words for episodes, 50–100 for 200-word sub-episodes). No speaker labels.
- For 2 speakers: Write a dialogue with named speakers (Host: Alex, Guest: Sam). Use 'SpeakerName: ' for each paragraph, continuous without extra newlines. Include Q&A where the host asks questions and the guest answers, with light humor or debate. Balance contributions.
- For 3 speakers: Write a dialogue with named speakers (Host: Alex, Guests: Sam, Taylor). Use 'SpeakerName: ' for each paragraph, continuous without extra newlines. Include Q&A with the host leading questions and both guests responding, with occasional debate or humor. Balance contributions.
**Episode/Sub-Episode Instructions**:
- Episode X (1000 words total, 3 sub-episodes):
  - Sub-Episode 1 (200 words): Introduce the episode’s theme with a unique opening. Use 1–2 paragraphs.
  - Sub-Episode 2 (600 words): Deepen the theme with detailed analysis/examples. Include news for Episode 3+. Use 3–4 paragraphs.
  - Sub-Episode 3 (200 words): Conclude the episode’s theme. Use 1–2 paragraphs.
**Context**:
- Topic: {topic}
- Description: {description}
- Episode: {episode_title}
- Sub-Episode: {sub_episode_title}
- Word Count: {word_count}
- Previous Content: {previous_content}
- Speakers: {speakers}
- News: {news}
- Location: {location}
- Total Episodes: {total_episodes}
- Episode Names: {episode_names}
Output the script as plain text with speaker labels (if 2+ speakers, continuous without extra newlines) and paragraph breaks, excluding the episode/sub-episode title. Ensure exactly {word_count} words and a complete final sentence.
"""

WRAPUP_PROMPT = """You are a world-class podcast scriptwriter crafting an engaging Wrap-up Session for a podcast series. Write a script that is **exactly 300 words** (no more, no less) for the given topic and description. The script must:
- Start with a reflective hook (e.g., for Remote Work: 'Reflecting on remote work’s rise…')
- Summarize all episodes, referencing their themes and key insights
- Integrate **recent news** (from {news}) if available to tie into the summary
- Incorporate **location-specific cultural references** (from {location}) to ground the script
- End with a **clear, conclusive outlook** on the topic’s future
- Include a thank-you to listeners and reference the topic name (e.g., 'Thank you for joining The Future of Remote Work')
- For 2+ speakers, include a thank-you to the guests by name (e.g., 'Thank you for connecting with us, Sam and Taylor')
- Maintain a conversational, authoritative, and engaging tone
- Use 2–3 paragraphs for readability
- End with a complete sentence
**Speaker Instructions**:
- For 1 speaker: Write a monologue with clear paragraphs (100–150 words). No speaker labels.
- For 2+ speakers: Write a dialogue with named speakers (Host: Alex, Guest(s): Sam, Taylor). Use 'SpeakerName: ' for each paragraph, continuous without extra newlines. The host leads the thank-you and conclusion, with guests adding brief reflections. Include light Q&A or banter. Balance contributions.
**Context**:
- Topic: {topic}
- Description: {description}
- Episode: Wrap-up Session
- Word Count: 300
- Previous Content: {previous_content}
- Speakers: {speakers}
- News: {news}
- Location: {location}
- Total Episodes: {total_episodes}
- Episode Names: {episode_names}
Output the script as plain text with speaker labels (if 2+ speakers, continuous without extra newlines) and paragraph breaks, excluding the episode title. Ensure exactly 300 words and a complete final sentence.
"""

def fetch_news(topic: str, location: str) -> str:
    try:
        query = f"{topic} 2025 {location}"
        news = google_news.get_news(query)
        news_items = [
            f"{item['title']} (Source: {item['publisher']['title']}, {item['published date']})"
            for item in news
        ]
        return "\n".join(news_items) if news_items else "No relevant news found."
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return "No relevant news found."

def calculate_structure(state: PodcastState) -> PodcastState:
    try:
        words_per_minute = 150
        total_words = state["duration"] * words_per_minute
        intro_words = 450
        wrapup_words = 300
        episode_words = 1000
        sub_episode_words = [200, 600, 200]
       
        remaining_words = total_words - intro_words - wrapup_words
        num_episodes = min(max(1, math.ceil(remaining_words / episode_words)), 10)
        total_episodes = num_episodes + 2
       
        prompt = EPISODE_NAMES_PROMPT.format(
            topic=state["topic"],
            description=state["description"],
            num_episodes=num_episodes
        )
        response = model.generate_content(prompt)
        names = [name.strip() for name in response.text.strip().split("\n") if name.strip()][:num_episodes]
        if len(names) < num_episodes:
            fallback_names = [
                "Core Insights", "Historical Context", "Modern Trends",
                "Innovative Solutions", "Global Perspectives", "Future Horizons"
            ]
            names.extend(fallback_names[:num_episodes - len(names)])

        structure = [
            {"episode": "Introduction", "duration": intro_words / words_per_minute, "word_count": intro_words, "sub_episodes": []}
        ]
        for i, name in enumerate(names):
            structure.append({
                "episode": f"Episode {i+1}: {name}",
                "duration": episode_words / words_per_minute,
                "word_count": episode_words,
                "sub_episodes": [
                    {"sub_episode": f"Sub-Episode {j+1}", "duration": w / words_per_minute, "word_count": w}
                    for j, w in enumerate(sub_episode_words)
                ]
            })
        structure.append({
            "episode": "Wrap-up Session",
            "duration": wrapup_words / words_per_minute,
            "word_count": wrapup_words,
            "sub_episodes": []
        })

        logger.info(f"Structure generated: {structure}")
        return {
            **state,
            "structure": structure,
            "total_episodes": total_episodes,
            "episode_names": names,
            "filename": ""
        }
    except Exception as e:
        logger.error(f"Error in calculate_structure: {e}")
        return {**state, "error": str(e), "filename": ""}

def fetch_news_node(state: PodcastState) -> PodcastState:
    news = fetch_news(state["topic"], state["location"])
    return {**state, "news": news}

def generate_scripts(state: PodcastState) -> PodcastState:
    scripts = []
    previous_content = ""
    speaker_names = ["Alex"] if state["speakers"] == 1 else ["Alex", "Sam"] if state["speakers"] == 2 else ["Alex", "Sam", "Taylor"]
    
    def generate_script_content(episode_title, sub_episode_title, target_word_count, episode_news):
        max_attempts = 3
        adjusted_word_count = int(target_word_count * 2.5)
        last_script = ""
        
        for attempt in range(max_attempts):
            try:
                if episode_title == "Introduction":
                    prompt_template = INTRODUCTION_PROMPT
                elif episode_title == "Wrap-up Session":
                    prompt_template = WRAPUP_PROMPT
                else:
                    prompt_template = EPISODE_PROMPT
                
                prompt = prompt_template.format(
                    topic=state["topic"],
                    description=state["description"],
                    episode_title=episode_title,
                    sub_episode_title=sub_episode_title,
                    word_count=adjusted_word_count,
                    previous_content=previous_content,
                    speakers=state["speakers"],
                    news=episode_news,
                    location=state["location"],
                    total_episodes=state["total_episodes"],
                    episode_names=", ".join(state["episode_names"])
                )
                response = model.generate_content(prompt)
                script = response.text.strip()
                
                sentences = re.split(r'(?<=[.!?])\s+', script)
                words = []
                current_word_count = 0
                
                for sentence in sentences:
                    sentence_words = sentence.split()
                    if current_word_count + len(sentence_words) <= target_word_count + 10:
                        words.extend(sentence_words)
                        current_word_count += len(sentence_words)
                    else:
                        break
                
                word_count = len(words)
                last_script = " ".join(words)
                
                if target_word_count - 10 <= word_count <= target_word_count:
                    if word_count > target_word_count:
                        words = words[:target_word_count]
                    elif word_count < target_word_count:
                        filler = "This topic continues to expand."
                        filler_words = filler.split()
                        words.extend(filler_words[:target_word_count - word_count])
                    return " ".join(words)
                adjusted_word_count = int(adjusted_word_count * 1.2)
            except Exception as e:
                logger.error(f"Error generating script for {episode_title}/{sub_episode_title}: {e}")
                last_script = f"Error: {str(e)}"
        
        words = last_script.split()[:target_word_count]
        return " ".join(words)

    for episode in state["structure"]:
        episode_title = episode["episode"]
        logger.info(f"Processing {episode_title}")
        
        if episode["sub_episodes"]:
            episode_script = ""
            episode_word_count = 0
            episode_sub_scripts = []
            
            for sub_episode in episode["sub_episodes"]:
                sub_episode_title = sub_episode["sub_episode"]
                target_word_count = sub_episode["word_count"]
                episode_news = state["news"] if (
                    episode_title == "Introduction" or
                    (episode_title.startswith("Episode 3") or 
                     (episode_title.startswith("Episode") and 
                      int(episode_title.split(":")[0].replace("Episode ", "")) > 3) and 
                     sub_episode_title == "Sub-Episode 2") and 
                    state["news"] != "No relevant news found."
                ) else ""
                
                script = generate_script_content(episode_title, sub_episode_title, target_word_count, episode_news)
                episode_script += script + "\n"
                episode_word_count += target_word_count
                episode_sub_scripts.append({
                    "sub_episode": sub_episode_title,
                    "script": script,
                    "word_count": target_word_count
                })
                previous_content = f"{episode_title}/{sub_episode_title}:\n{script}\n"
                time.sleep(0.3)
            scripts.append({
                "episode": episode_title,
                "sub_episode": None,
                "script": episode_script.strip(),
                "word_count": episode_word_count,
                "sub_scripts": episode_sub_scripts
            })
        else:
            target_word_count = episode["word_count"]
            episode_news = state["news"] if (
                episode_title == "Introduction" or 
                (episode_title == "Wrap-up Session" and state["news"] != "No relevant news found.")
            ) else ""
            
            script = generate_script_content(episode_title, episode_title, target_word_count, episode_news)
            scripts.append({
                "episode": episode_title,
                "sub_episode": None,
                "script": script,
                "word_count": target_word_count,
                "sub_scripts": []
            })
            previous_content = f"{episode_title}:\n{script}\n"
            time.sleep(0.1)
    
    return {**state, "scripts": scripts, "previous_content": previous_content}

def save_podcast(state: PodcastState) -> PodcastState:
    try:
        podcast_data = {
            "topic": state["topic"],
            "description": state["description"],
            "total_duration": state["duration"],
            "speakers": state["speakers"],
            "location": state["location"],
            "total_episodes": state["total_episodes"],
            "episode_names": state["episode_names"],
            "timestamp": datetime.now().isoformat(),
            "episodes": [
                {
                    "title": e["episode"],
                    "sub_episode": s["sub_episode"],
                    "duration": e["duration"],
                    "script": s["script"],
                    "word_count": s["word_count"],
                    "sub_scripts": s.get("sub_scripts", [])
                }
                for e, s in zip(state["structure"], state["scripts"])
            ]
        }
        filename = f"podcast_{state['topic'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(podcast_data, f, indent=2)
        logger.info(f"Saved podcast to {filename}")
        return {**state, "filename": filename}
    except Exception as e:
        logger.error(f"Error saving podcast: {e}")
        return {**state, "error": str(e), "filename": ""}

def generate_podcast(topic: str, description: str, duration: int, speakers: int, location: str) -> Dict:
    workflow = StateGraph(PodcastState)
    workflow.add_node("calculate_structure", calculate_structure)
    workflow.add_node("fetch_news", fetch_news_node)
    workflow.add_node("generate_scripts", generate_scripts)
    workflow.add_node("save_podcast", save_podcast)

    workflow.set_entry_point("calculate_structure")
    workflow.add_edge("calculate_structure", "fetch_news")
    workflow.add_edge("fetch_news", "generate_scripts")
    workflow.add_edge("generate_scripts", "save_podcast")
    workflow.add_edge("save_podcast", END)

    app = workflow.compile()

    initial_state = {
        "topic": topic,
        "description": description,
        "duration": duration,
        "speakers": speakers,
        "location": location,
        "structure": [],
        "total_episodes": 0,
        "episode_names": [],
        "news": "",
        "scripts": [],
        "previous_content": "",
        "error": "",
        "filename": ""
    }

    return app.invoke(initial_state)
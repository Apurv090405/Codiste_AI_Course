import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to get response from Gemini
def get_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# List of prompts (with Indian tone)
prompts = [
    """Act as a helpful assistant. Generate a list of three made-up book titles with their authors and genres in JSON format.""",

    """Act as a marketing assistant. Write a product description for a chair based on the following specs, focusing on materials for furniture retailers. \
Format as HTML with a dimensions table.
Specs: Material - oak wood,
leather upholstery;
Product ID - CH12345; 
Dimensions - height: 40in, width: 24in, depth: 24in.""",

    """Act as a review summarizer. Summarize the following review in at most 30 words.
Review: The cornfield is burning. The fire department is better off bringing butter and pinstripe paper cups. Your father claims, “Act of God.” There will be insurance money—maybe more than what the harvest would’ve brought in. Not enough to send you away to college. Your house is a mailbox cherry bomb. You can’t stay. Michigan is a mouth that only knows how to chew. Maybe a war in the desert is your destiny. You and I started our own religion in that field. Look at the fire hose baptism. Did you set the fire to see if I would stay?""",

    """Act as a translator. Translate the following English text to Gujarati (formal and informal forms): \
Text: Would you like to order a coffee?""",

    """Act as a customer service assistant. Reply to this review in a professional tone. \
If positive, thank them; if negative, apologize and offer support.
Review: This place has hands down the best brunch in town! The avocado toast with poached eggs was incredible, and their freshly squeezed orange juice was the perfect complement. The restaurant has such a bright, cheerful atmosphere with friendly staff who make you feel right at home. I've been here three weekends in a row and plan to keep coming back. Don't miss their homemade pastries - they're to die for!
""",

    """Act as a friendly chatbot named OrderBot for a pizza restaurant. Greet the user and ask for their order. \
Menu:
1. Margherita Pizza (₹499)
2. Pepperoni Pizza (₹675)
3. Paneer Tikka Pizza (₹625)
4. Veggie Overload Pizza (₹599)
5. Chicken BBQ Pizza (₹725)
6. Garlic Breadsticks (₹199)
7. Cheese Garlic Bread (₹249)
8. Coke (₹65)
9. Masala Lemonade (₹55)
10. Choco Lava Cake (₹149)"""
]

# Run all prompts and print responses
for i, prompt in enumerate(prompts, start=1):
    print(f"\n--- Prompt {i} ---\n")
    print(prompt.strip())
    print("\nResponse:")
    response = get_gemini_response(prompt)
    print(response)
    print("\n" + "-" * 100)
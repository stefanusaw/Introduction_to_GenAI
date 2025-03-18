from dotenv import load_dotenv

import os

load_dotenv()

try:
    import openai
    OPENAI_AVAILABLE = True
except ModuleNotFoundError:
    OPENAI_AVAILABLE = False

def get_api_key():
    """Retrieve API key from environment variable."""
    return os.getenv("OPENAI_API_KEY")

def generate_text(prompt, max_tokens=100, temperature=0.7):
    """Send a prompt to the AI model and return the response."""
    if not OPENAI_AVAILABLE:
        return "Error: The 'openai' module is not installed. Please install it using 'pip install openai'."
    
    api_key = get_api_key()
    if not api_key:
        return "Error: API key is missing. Set the OPENAI_API_KEY environment variable."

    # If using a custom API, configure `api_base`
    openai.api_key = api_key
    openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.aimlapi.com/v1")  # Default to OpenAI's official API

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, #length
            temperature=temperature #randomness a.k.a. creativity
        )
        return response["choices"][0]["message"]["content"].strip()
    except openai.error.AuthenticationError:
        return "Error: Invalid API key. Check your OpenAI API key."
    except openai.error.RateLimitError:
        return "Error: Rate limit exceeded. Check your OpenAI usage and plan."
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Simulate user input for environments without interactive input."""
    print("AI-Powered Text Completion")
    test_prompts = [
        "Write a short poem about nature.",
        "Summarize the importance of AI in healthcare.",
        "Explain quantum computing in simple terms."
        # "Write a short poem about work.",
        # "Summarize the importance of AI in education.",
        # "Explain Machine learning in simple terms."
    ]
    with open('output.txt', 'a') as file:
        for prompt in test_prompts:
            file.write(f"User Prompt: {prompt}\n")
            response = generate_text(prompt)
            file.write(f"AI Response: {response}\n\n")

if __name__ == "__main__":
    main()

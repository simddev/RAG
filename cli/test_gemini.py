import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")


def main():
    client = genai.Client(api_key=api_key)
    model = "gemma-4-31b-it"
    prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

    response = client.models.generate_content(model=model, contents=prompt)
    assert response.usage_metadata is not None

    print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    print(response.text)


if __name__ == "__main__":
    main()


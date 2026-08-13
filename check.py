import os
from google import genai

# Put your Gemini API key in the GEMINI_API_KEY environment variable
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY is not set.")
    print("Set your API key first.")
    exit(1)

try:
    client = genai.Client(api_key=api_key)

    print("=" * 70)
    print("AVAILABLE GEMINI MODELS")
    print("=" * 70)

    found = False

    for model in client.models.list():
        # Only show models that can generate content
        if "generateContent" in model.supported_actions:
            found = True
            print(f"\nModel: {model.name}")
            print(f"Display name: {model.display_name}")
            print(f"Input limit: {model.input_token_limit}")
            print(f"Output limit: {model.output_token_limit}")
            print(f"Supported actions: {model.supported_actions}")

    if not found:
        print("\nNo models supporting generateContent were found.")

except Exception as e:
    print("\nERROR:")
    print(e)
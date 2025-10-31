import google.generativeai as genai
import os

try:
    # --- IMPORTANT ---
    # This script assumes your API key is set as an 
    # environment variable named GOOGLE_API_KEY
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    print("Checking for models your API key can access...\n")
    
    # List all models
    for model in genai.list_models():
      # Check if the model supports the 'generateContent' method
      if 'generateContent' in model.supported_generation_methods:
        print(f"- {model.name}")

    print("\nCheck this list for 'gemini-1.5-pro' or 'gemini-1.5-pro-latest'")

except KeyError:
    print("--- ERROR ---")
    print("The 'GOOGLE_API_KEY' environment variable is not set.")
    print("Please set it in your terminal before running this script.")
except Exception as e:
    print(f"An error occurred: {e}")
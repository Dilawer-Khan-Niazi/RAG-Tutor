import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Configure API
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("🔍 Available Gemini Models:")
print("="*60)

# List all available models
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Description: {model.description[:100]}...")
        print()

print("="*60)
print("\n🧪 Testing a simple generation:")

# Test with the first available model
try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Say 'Hello! API is working!'")
    print(f"✅ Success: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")
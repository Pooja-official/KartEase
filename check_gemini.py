import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

print(f"Testing Gemini API with model: {model_name}")

llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key,
    temperature=0
)

response = llm.invoke("Hello, are you ready to assist KartEase customers?")
print("\n--- Milestone 1 Passed! Response from Gemini ---")
print(response.content)
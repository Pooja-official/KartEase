import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tools import get_order_status, search_policies, extract_text

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
chat_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

tools = [get_order_status, search_policies]

system_prompt = (
    "You are the official KartEase customer support assistant.\n"
    "Your job is to answer customer questions about KartEase orders and policies.\n"
    "Rules:\n"
    "- Always use 'get_order_status' to fetch details for order inquiries.\n"
    "- Always use 'search_policies' for any policy, shipping, payment, return, or warranty questions.\n"
    "- If a query requires both (e.g., checking an order's status/payment and refund policy), invoke BOTH tools.\n"
    "- If a question is unrelated to KartEase (e.g., general knowledge, math, poems), do not answer it. Respond ONLY with:\n"
    "'Sorry, I can only help with KartEase orders and policies.'\n"
    "- Do not invent facts."
)

llm = ChatGoogleGenerativeAI(
    model=chat_model,
    google_api_key=api_key,
    temperature=0
)

agent = create_agent(llm, tools, system_prompt=system_prompt)

def ask(question: str):
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = response["messages"]
    
    # Identify tools used
    tools_used = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc.get("name"))
        elif hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs:
            for tc in msg.additional_kwargs["tool_calls"]:
                tools_used.append(tc.get("function", {}).get("name"))
                
    unique_tools = list(dict.fromkeys(tools_used))
    tool_label = f"[Tool(s) used: {', '.join(unique_tools)}]" if unique_tools else "[Tool(s) used: None]"
    
    last_content = extract_text(messages[-1].content)
    return last_content, tool_label

def main():
    print("=" * 60)
    print("KartEase Support Assistant (Terminal Chat)")
    print("Type 'exit' to quit.")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nCustomer: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Thank you for contacting KartEase. Goodbye!")
                break
                
            reply, tools_label = ask(user_input)
            print(f"{tools_label}")
            print(f"Agent: {reply}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session...")
            break

if __name__ == "__main__":
    main()
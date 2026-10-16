import os
import csv
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
embed_model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
chat_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Helper function to extract plain text from any AIMessage content
def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)

# ============================================================
# 1. Plain Order Status Function (R3 - Tested by pytest)
# ============================================================
def get_order_status_logic(order_id: str) -> str:
    """Reads orders.csv and returns details, case-insensitively."""
    target_id = order_id.strip().upper()
    csv_file = "orders.csv"
    
    if not os.path.exists(csv_file):
        return "Error: orders.csv not found."
    
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["order_id"].strip().upper() == target_id:
                return (
                    f"Order ID: {row['order_id']}, Product: {row['product']}, "
                    f"Status: {row['status']}, Date: {row['expected_or_delivered_date']}, "
                    f"Payment: {row['payment_method']}, Amount: ₹{row['amount_inr']}"
                )
    return f"No order found with ID {order_id}."

# ============================================================
# 2. Plain Policy Search Function (R2 - Milestone 3)
# ============================================================
def search_policies_logic(question: str) -> str:
    """Searches Chroma vector store and answers strictly from policy chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embed_model,
        google_api_key=api_key
    )
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    docs = vector_store.similarity_search(question, k=3)
    
    if not docs:
        return "Sorry, I can only help with KartEase orders and policies."

    # Format context blocks and source file names
    context_blocks = []
    for d in docs:
        source_name = os.path.basename(d.metadata.get("source", "policy document"))
        context_blocks.append(f"[{source_name}]: {d.page_content}")
    context_text = "\n\n".join(context_blocks)
    
    llm = ChatGoogleGenerativeAI(
        model=chat_model,
        google_api_key=api_key,
        temperature=0
    )
    
    prompt = (
        "You are the KartEase Support Agent. Answer the question using ONLY the provided context.\n"
        "Rules:\n"
        "1. Keep your answer concise and factual.\n"
        "2. Explicitly cite the source file name (e.g. [returns_and_refunds.md]).\n"
        "3. If the context does not contain the answer, reply EXACTLY:\n"
        "'Sorry, I can only help with KartEase orders and policies.'\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    res = llm.invoke(prompt)
    return extract_text(res.content).strip()

# ============================================================
# 3. LangChain Tools for the Agent (R4)
# ============================================================
@tool
def get_order_status(order_id: str) -> str:
    """Look up order details such as product name, delivery status, expected/delivered date, and payment method for a specific KartEase order ID (e.g., KE1001)."""
    return get_order_status_logic(order_id)

@tool
def search_policies(question: str) -> str:
    """Search KartEase store policies including returns, refund timelines, delivery charges, shipping times, warranty, and payment methods."""
    return search_policies_logic(question)
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tools import get_order_status, search_policies, extract_text

# Load configurations
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
chat_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Streamlit Page Config
st.set_page_config(
    page_title="KartEase Support Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #ff4b4b;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Cache Agent initialization
@st.cache_resource
def load_agent():
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
    return create_agent(llm, tools, system_prompt=system_prompt)

agent = load_agent()

# Function to execute query through Agent
def run_agent(query_text):
    with st.spinner("Processing intent with Gemini Agent..."):
        try:
            response = agent.invoke({"messages": [{"role": "user", "content": query_text}]})
            all_msgs = response["messages"]

            tools_used = []
            for m in all_msgs:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        tools_used.append(tc.get("name"))
                elif hasattr(m, "additional_kwargs") and "tool_calls" in m.additional_kwargs:
                    for tc in m.additional_kwargs["tool_calls"]:
                        tools_used.append(tc.get("function", {}).get("name"))

            unique_tools = list(dict.fromkeys(tools_used))
            final_reply = extract_text(all_msgs[-1].content)
            return final_reply, unique_tools
        except Exception as e:
            return f"Error executing agent: {str(e)}", []

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("🟢 Agent Active")
    st.info(f"**Model:** `{chat_model}`")
    st.info(f"**Embeddings:** `gemini-embedding-001`")

    st.markdown("---")
    st.subheader("📁 Data Explorer")
    
    # Orders CSV inspection
    if os.path.exists("orders.csv"):
        if st.checkbox("View `orders.csv` Records"):
            df_orders = pd.read_csv("orders.csv")
            st.dataframe(df_orders[["order_id", "product", "status", "payment_method"]], height=200)

    # Policy files inspection
    if os.path.exists("data"):
        policies = [f for f in os.listdir("data") if f.endswith(".md")]
        if st.checkbox("View Available Policy Files"):
            st.write(policies)

# ----------------- MAIN DASHBOARD -----------------
st.title("🛍️ KartEase AI Support Agent Dashboard")
st.markdown("Autonomous multi-tool assistant routing between **Vector Store RAG** and **Orders Database**.")

# Metric Overview Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Policy Documents", "4 Files", delta="ChromaDB")
with col2:
    orders_count = len(pd.read_csv("orders.csv")) if os.path.exists("orders.csv") else 0
    st.metric("Indexed Orders", f"{orders_count} Records", delta="orders.csv")
with col3:
    st.metric("Tools Configured", "2 Tools", delta="Orders + RAG")
with col4:
    st.metric("Architecture", "Hybrid RAG", delta="LangChain")

st.markdown("---")

# Layout: Split into Question Picker (Left) and Live Response Screen (Right)
left_col, right_col = st.columns([1, 1.2])

# Benchmark Questions Catalog
benchmark_categories = {
    "📜 Policy & RAG Questions": [
        "How many days do I have to return a phone?",
        "Can I return earbuds if I just don't like them?",
        "What is the delivery charge on a ₹350 order?",
        "If I order at 4 PM, when will it be dispatched?",
        "Can I use a coupon together with a bank offer?",
        "How long is Extended Protection and when can I buy it?",
        "Can I pay cash on delivery for a ₹12,000 order?"
    ],
    "📦 Order Lookup Questions": [
        "Where is my order KE1002?",
        "What is the status of order ke1005?",
        "What is the status of order KE9999?"
    ],
    "🔀 Multi-Tool Compound Questions": [
        "My return for KE1006 was picked up. When will I get my refund?",
        "I paid cash for KE1009. If I return it, how do I get my money back?"
    ],
    "🚫 Out-of-Scope (Guardrails)": [
        "Who is the CEO of Google?",
        "Write me a poem about Diwali."
    ]
}

selected_query = None

with left_col:
    st.subheader("📋 Select Benchmark Question")
    
    # Dropdown by category
    selected_cat = st.selectbox("Filter category:", list(benchmark_categories.keys()))
    questions_in_cat = benchmark_categories[selected_cat]
    
    chosen_from_dropdown = st.selectbox("Select question to evaluate:", questions_in_cat)
    if st.button("🚀 Ask Selected Question", key="btn_dropdown"):
        selected_query = chosen_from_dropdown

    st.markdown("#### Or click a quick benchmark:")
    # Render quick click buttons
    for q in questions_in_cat[:3]:
        if st.button(f"👉 {q}", key=f"quick_{q}"):
            selected_query = q

    st.markdown("---")
    custom_q = st.text_input("Or enter your own custom question:")
    if st.button("Submit Custom Question") and custom_q.strip():
        selected_query = custom_q.strip()

with right_col:
    st.subheader("🤖 Agent Reasoning & Output")
    
    if selected_query:
        st.markdown(f"**Customer Query:** `{selected_query}`")
        
        reply, tools = run_agent(selected_query)
        
        # Tool badge rendering
        if tools:
            st.success(f"🛠️ **Tools Executed:** `{'`, `'.join(tools)}`")
        else:
            st.warning("🛡️ **Tools Executed:** `None` (Guardrail Triggered / Handled directly)")

        # Display final response in a styled card
        st.markdown("### Answer:")
        st.info(reply)
        
        # Details expander for viva demonstration
        with st.expander("🔍 Trace Details & Viva Explanation"):
            if "search_policies" in tools and "get_order_status" in tools:
                st.write("**Routing Strategy:** Multi-Tool Call. The model fetched order payment metadata from `orders.csv` and cross-referenced refund timelines in `returns_and_refunds.md`.")
            elif "search_policies" in tools:
                st.write("**Routing Strategy:** Semantic RAG. Retrieved top-3 chunk vectors from ChromaDB and synthesized answer with source citation.")
            elif "get_order_status" in tools:
                st.write("**Routing Strategy:** Deterministic Tool. Extracted the Order ID parameter and retrieved the row from `orders.csv`.")
            else:
                st.write("**Routing Strategy:** Guardrail. Model detected the prompt is off-domain and replied with the strict boundary message.")
    else:
        st.info("👈 Select a question on the left panel or click a quick benchmark to run the agent live.")
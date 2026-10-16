# KartEase Support Agent

An intelligent, autonomous customer support assistant built for **KartEase**, a fictional Indian e-commerce platform. The assistant handles customer inquiries by intelligently deciding whether to query company policy documents via **Retrieval-Augmented Generation (RAG)**, fetch live order details via **Tool Calling**, invoke both tools, or politely decline out-of-scope questions.

---

## 1. Project Overview & Architecture

KartEase customer service frequently answers repetitive queries regarding returns, shipping fees, warranty terms, and order tracking. This system automates those workflows:

- **Policy Knowledge Base (RAG):** Answers store policy queries strictly from four markdown policy documents (`returns_and_refunds.md`, `shipping_and_delivery.md`, `warranty_and_repairs.md`, `payments_and_offers.md`) stored as vector embeddings in a persistent Chroma database.
- **Order Lookup Tool:** Reads live order data from `orders.csv` and returns order details (product, status, date, payment method) case-insensitively.
- **Autonomous Agent:** Powered by Google Gemini (`gemini-3.8-flash`), the agent evaluates incoming user intent, determines which tool(s) to invoke, synthesizes tool outputs into a clear response, cites source policy files, and rejects unrelated prompts.

---

## 2. Framework Choice: Why LangChain?

We selected **LangChain** (`langchain`, `langchain-google-genai`, `langchain-chroma`) over LlamaIndex for this project:

1. **Synchronous Execution:** LangChain agents invoke cleanly with standard synchronous execution (`agent.invoke(...)`) without requiring custom asynchronous event loop wrappers.
2. **Native Tool Calling:** Function tools are defined cleanly with Python's `@tool` decorator, directly inspecting function type annotations and docstrings for agent reasoning.
3. **Seamless Multi-Tool Routing:** LangChain handles single-turn reasoning, multi-tool parallel calls (combining order lookup with refund policy lookups), and fallback responses out of the box.

---

## 3. Project Structure

```text
kartease_starter/
├── data/
│   ├── payments_and_offers.md
│   ├── returns_and_refunds.md
│   ├── shipping_and_delivery.md
│   └── warranty_and_repairs.md
├── tests/
│   └── test_order_tool.py
├── .env.example
├── .gitignore
├── app.py
├── check_gemini.py
├── ingestion.py
├── orders.csv
├── pytest.ini
├── README.md
└── requirements-langchain.txt
# Local-RAG-Assistant
-> Intelligent Policy Assistant

### A Local, Privacy-First RAG Application for Banking Compliance

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Stack](https://img.shields.io/badge/Tech-LangChain%20|%20Streamlit%20|%20Ollama-green)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

**DocuMind** is an AI-powered assistant designed to navigate complex, high-stakes documentation (such as Transaction Processing Rules). Unlike standard cloud-based chatbots, this system runs **entirely locally** using Llama 3 and ChromaDB, ensuring that sensitive financial data never leaves the secure environment.

---

## 🏗️ Architecture

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to ground AI responses in strict factual data.



1.  **Ingestion:** The system loads a 400+ page PDF (`transaction-processing-rules.pdf`).
2.  **Indexing:** Text is chunked and embedded using **HuggingFace Embeddings** and stored in a persistent **ChromaDB** vector store.
3.  **Retrieval:** When a user asks a question, the system performs a semantic search to find the top 3 relevant pages.
4.  **Generation:** The retrieved context + user query are sent to **Llama 3 (via Ollama)** to generate a precise, cited answer.
5.  **Interface:** A clean, chat-based UI built with **Streamlit**.

---

## 🛠️ Tech Stack

* **LLM Engine:** [Ollama](https://ollama.com/) (running Llama 3 8B)
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB (Local Persistence)
* **Embeddings:** `all-MiniLM-L6-v2` (HuggingFace)
* **Frontend:** Streamlit
* **Language:** Python 3.10+

---

## 🚀 Key Features

* **🔒 Zero Data Leakage:** Runs 100% offline. No API keys sent to OpenAI/Google. Perfect for BFSI (Banking, Financial Services, Insurance) sectors.
* **🧠 Context-Aware:** Uses semantic search to understand the *meaning* of a query, not just keyword matching.
* **⚡ Cost Efficient:** No per-token API costs. Runs on consumer hardware (tested on 16GB RAM Laptop).
* **🛡️ Citation:** The AI cites the specific Rule Number or Page Number it used to generate the answer.

---

## 💻 Installation & Setup

### 1. Prerequisites
* Python 3.9 or higher installed.
* **Ollama** installed and running.
    ```bash
    # Pull the Llama 3 model
    ollama pull llama3
    ```

### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/DocuMind-Compliance-RAG.git](https://github.com/your-username/DocuMind-Compliance-RAG.git)
cd DocuMind-Compliance-RAG
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
 ```
### 4. Build the Knowledge Base (One-Time Setup)
This script parses the PDF and creates the Vector Database.
```bash
python step1_retriever_local.py
 ```

### 5. Launch the Application
Start the web interface.
```bash
streamlit run step3_ui.py
 ```
The app will open automatically at http://localhost:8501

![UI](UI_interface.png)

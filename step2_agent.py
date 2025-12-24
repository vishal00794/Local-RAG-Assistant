import sys
import os
from dotenv import load_dotenv

# --- IMPORTS ---
# We use the community Ollama integration
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import the Librarian from Step 1
try:
    from step1_retriever_local import PolicyLibrarian
except ImportError:
    print("❌ Critical Error: Could not import 'step1_retriever_local.py'.")
    print("Make sure both files are in the same folder.")
    sys.exit(1)

# ---------------------------------------------------------
# THE LOCAL INTELLIGENT AGENT
# ---------------------------------------------------------
class BankComplianceAgent:
    def __init__(self):
        load_dotenv()
        
        print("🦙 Connecting to Local Ollama (Llama 3)...")
        
        # 1. Initialize Local LLM
        # This connects to your locally running Ollama instance (usually port 11434)
        self.llm = ChatOllama(
            model="llama3",
            temperature=0.0, # Strict facts only
            keep_alive="1h"  # Keeps the model loaded in RAM for faster follow-up queries
        )

        # 2. Connect to the Knowledge Base (Disk)
        print("📚 Connecting to Policy Database (Disk)...")
        self.librarian = PolicyLibrarian()

        # 3. Define the Persona
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are a Compliance Officer at SentinAL Bank. 
        You are answering queries based strictly on the provided internal policy documents.
        
        INSTRUCTIONS:
        1.  **Cite your sources**: Mention the specific Rule Number (e.g., "Rule 1.3.2") or Page Number.
        2.  **Be precise**: If the user asks about a specific region (like Canada), look for regional exceptions.
        3.  **No guessing**: If the answer is not in the context, explicitly state: "I cannot find this information in the official policy documents."
        4.  **Tone**: Professional, authoritative, and direct, easy wordings for non-experts.

        CONTEXT (Retrieved from Database):
        {context}

        USER QUESTION: 
        {question}

        YOUR RESPONSE:
        """)

    def format_docs(self, docs):
        """Prepares the retrieved documents for the AI to read."""
        formatted_text = ""
        for doc in docs:
            source = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content.replace("\n", " ")
            formatted_text += f"--- DOCUMENT START ---\nSOURCE: {source} (Page {page})\nCONTENT: {content}\n--- DOCUMENT END ---\n\n"
        return formatted_text

    def ask(self, query: str):
        print(f"\n💬 Question: {query}")
        print("   🔍 Searching internal database...")
        
        # 1. RETRIEVE
        docs = self.librarian.search(query)
        
        if not docs:
            return "❌ I searched the database but could not find any relevant documents regarding this topic."

        # 2. AUGMENT
        context_text = self.format_docs(docs)
        
        # 3. GENERATE
        print("   🧠 Thinking (Local Llama 3)...")
        chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        # We use a try/except block to catch connection errors if Ollama isn't running
        try:
            response = chain.invoke({"context": context_text, "question": query})
            return response
        except Exception as e:
            return f"❌ Ollama Error: Is Ollama running? Run 'ollama serve' in a separate terminal. Details: {e}"

# ---------------------------------------------------------
# MAIN EXECUTION LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        agent = BankComplianceAgent()
        
        print("\n" + "="*60)
        print("🛡️  SentinAL COMPLIANCE AGENT READY")
        print("   (Running LOCALLY on Ollama/Llama3 - No Internet Required)")
        print("="*60)
        
        # Interactive Loop
        while True:
            user_query = input("\n📝 Enter Question (or 'exit'): ")
            if user_query.lower() in ['exit', 'quit']:
                print("👋 Exiting system.")
                break
            
            if not user_query.strip():
                continue

            response = agent.ask(user_query)
            
            print("\n" + "-"*60)
            print(response)
            print("-" * 60)
            
    except Exception as e:
        print(f"\n❌ System Error: {e}")
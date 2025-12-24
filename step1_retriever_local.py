import os
import shutil
import glob
import pickle
from typing import List, Sequence, Tuple, Optional, Iterator
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# We import the BaseStore to create our own robust storage
from langchain_core.stores import BaseStore

try:
    from langchain.retrievers import ParentDocumentRetriever
except ImportError:
    from langchain_classic.retrievers import ParentDocumentRetriever

# ---------------------------------------------------------
# HELPER: Custom Disk Store (Fixes the Import Error)
# ---------------------------------------------------------
class SimpleDiskStore(BaseStore[str, Document]):
    """
    A simple, robust storage that saves Documents to disk using pickle.
    This replaces LocalFileStore and works in ANY environment.
    """
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def _get_path(self, key: str) -> str:
        # Use a hash of the key to ensure safe filenames
        import hashlib
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.folder_path, f"{safe_key}.pkl")

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        results = []
        for key in keys:
            path = self._get_path(key)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        results.append(pickle.load(f))
                except Exception:
                    results.append(None)
            else:
                results.append(None)
        return results

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        for key, doc in key_value_pairs:
            path = self._get_path(key)
            with open(path, "wb") as f:
                pickle.dump(doc, f)

    def mdelete(self, keys: Sequence[str]) -> None:
        for key in keys:
            path = self._get_path(key)
            if os.path.exists(path):
                os.remove(path)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        # Not strictly needed for retrieval, but required by BaseClass
        yield from []

# ---------------------------------------------------------
# CORE LOGIC: The Persistent Librarian
# ---------------------------------------------------------
class PolicyLibrarian:
    def __init__(self, persist_directory="./chroma_db", fs_directory="./parent_docs_store"):
        load_dotenv()
        
        print("🔌 Loading Local Embeddings (HuggingFace)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 1. Vectorstore (ChromaDB - Disk)
        self.vectorstore = Chroma(
            collection_name="mastercard_rules_db_final", 
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # 2. Docstore (Our Custom DiskStore)
        # This saves the full text to the './parent_docs_store' folder
        self.store = SimpleDiskStore(fs_directory)
        
        # Child: 100 chars (Searchable vectors)
        # Parent: 3000 chars (Full context)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

    def ingest_folder(self, folder_path: str):
        print(f"\n📂 Scanning folder: {folder_path}...")
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDFs found in '{folder_path}'!")
            return

        for pdf_path in pdf_files:
            try:
                print(f"\n   📖 Reading: {os.path.basename(pdf_path)}...")
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata["source_file"] = os.path.basename(pdf_path)

                total_pages = len(docs)
                print(f"      Found {total_pages} pages. Starting Batched Ingestion...")

                # Process 50 pages at a time to avoid crashing ChromaDB
                BATCH_SIZE = 50 
                for i in range(0, total_pages, BATCH_SIZE):
                    batch = docs[i : i + BATCH_SIZE]
                    print(f"      Processing pages {i+1} to {min(i+BATCH_SIZE, total_pages)}...")
                    self.retriever.add_documents(batch, ids=None)
                
                print(f"     ✅ Successfully Indexed {os.path.basename(pdf_path)}")
                
            except Exception as e:
                print(f"     ❌ Failed to load {pdf_path}: {e}")
        
        print("\n✅ Ingestion Complete! Database is ready.")

    def search(self, query: str) -> List[Document]:
        print(f"\n🔍 Searching for: '{query}'")
        results = self.retriever.invoke(query)
        return results

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Clean up old DBs to force a clean start with the new Storage Class
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    if os.path.exists("./parent_docs_store"):
        shutil.rmtree("./parent_docs_store")
    print("🧹 Cleaned up old databases.")

    # 2. Initialize
    librarian = PolicyLibrarian()
    
    # 3. Ingest
    librarian.ingest_folder("data_docs")
    
    # 4. Interactive Test
    print("\n" + "="*40)
    print("🤖 SENTINAL READY. Type 'exit' to quit.")
    print("="*40)
    
    while True:
        user_query = input("\nEnter Query: ")
        if user_query.lower() in ['exit', 'quit']:
            break
            
        retrieved_docs = librarian.search(user_query)
        
        if not retrieved_docs:
            print("❌ No matching documents found.")
        else:
            print(f"\n--- 📄 FOUND {len(retrieved_docs)} RESULTS ---")
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source_file', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                print(f"\nResult {i+1} (Source: {source}, Page: {page}):")
                print(doc.page_content.replace('\n', ' ')[:400] + "...") 
                print("-" * 40)
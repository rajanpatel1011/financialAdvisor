import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# Directory where your financial policy and knowledge files are stored
DATA_PATH = "finance_knowledge_base/" 
# Path for the local vector database
CHROMA_DB_PATH = "chroma_db_financial_advice"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def create_knowledge_base():
    """
    Loads documents, chunks them, creates embeddings, and saves them to ChromaDB.
    This should be run whenever the source documents change.
    """
    print("--- 🛠️ STEP 1: Knowledge Base Creation/Ingestion ---")
    
    # 1. Load Documents
    print(f"Loading documents from: {DATA_PATH}")
    # Using TextLoader for .txt files. Adjust glob/loader for PDFs, JSON, etc.
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)
    documents = loader.load()

    if not documents:
        print("🛑 No documents loaded. Check DATA_PATH and file types.")
        return False

    # 2. Split Documents into Chunks
    print(f"Splitting {len(documents)} document(s) into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Create Embeddings and Store
    print(f"Embedding chunks and saving to ChromaDB at: {CHROMA_DB_PATH}")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory=CHROMA_DB_PATH
        )
        db.persist()
        print("✅ Knowledge base created successfully.")
        return True
    except Exception as e:
        print(f"❌ Error during embedding or database creation: {e}")
        return False

def retrieve_financial_reference(user_query: str, k: int = 4):
    """
    Pulls the most relevant chunks of financial knowledge for a given query from the DB.
    """
    print("\n--- 🔎 STEP 2: Retrieval ---")
    
    # Check if the database exists; if not, create it
    if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
        print(f"Database not found at {CHROMA_DB_PATH}. Creating it now...")
        if not create_knowledge_base():
            return ["Error: Cannot retrieve reference material because the database could not be created."]

    try:
        # 1. Load the Vector Database
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(
            persist_directory=CHROMA_DB_PATH, 
            embedding_function=embeddings
        )

        # 2. Perform Semantic Search
        retriever = db.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(user_query)
        
        # 3. Format the Reference Output
        reference_material = [
            doc.page_content for doc in relevant_docs
        ]
        
        print(f"Retrieved {len(reference_material)} relevant document chunks.")
        
        return reference_material

    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        return ["Error during retrieval: Please check configuration and API key."]

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure your API key is set in your environment
    if 'OPENAI_API_KEY' not in os.environ:
        print("🚨 Warning: OPENAI_API_KEY environment variable is not set. Retrieval will likely fail.")

    # Example Query
    query = "Summarize the latest Q3 investment guidelines for fixed-income assets as per the compliance memo."
    
    retrieved_reference = retrieve_financial_reference(query, k=3)
    
    print("\n--- Final Output: Context for Gemini ---")
    print(f"User Query: {query}")
    print("---------------------------------------")
    
    if "Error" not in retrieved_reference[0]:
        for i, ref in enumerate(retrieved_reference):
            print(f"Chunk {i+1} (Source Context):")
            print(f">>> {ref[:300]}...\n")
    else:
        print(retrieved_reference[0])

    # This 'retrieved_reference' list would be sent to the Gemini API 
    # as grounded context for generating a final, accurate financial advice response.
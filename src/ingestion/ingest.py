import os
import glob
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import TextLoader, PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_community.embeddings import FastEmbedEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS

def main():
    # Load environment variables
    load_dotenv()
    
    # Define directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data", "sample_policies")
    persist_directory = os.path.join(base_dir, "faiss_index")
    
    documents = []
    
    # 1. Read PDF or TXT policy documents
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return
        
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
    if not documents:
        print("No documents found in data/sample_policies/")
        return
        
    # 2. Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Embed using HuggingFace sentence-transformers
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    # 4. Store in local FAISS vector store
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    vectorstore.save_local(persist_directory)
    
    print(f"Successfully processed {len(documents)} documents.")
    print(f"Stored {len(chunks)} chunks into FAISS vector store at {persist_directory}")

if __name__ == "__main__":
    main()

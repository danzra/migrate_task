from fastapi import FastAPI, BackgroundTasks, File, UploadFile
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pathlib import Path
from pydantic import BaseModel  # For creating request body models

# FastAPI instance
app = FastAPI()

UPLOAD_DIRECTORY = "fastapi_2/uploaded_files"
INDEX_DIRECTORY = "fastapi_2/faiss_index"  # Path to where FAISS index is saved

# Set API Key
os.environ["GROQ_API_KEY"] = "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"

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.environ["GROQ_API_KEY"]
)

# Function to extract text from PDF
def extract_text_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    documents = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            documents.append(
                Document(page_content=text, metadata={"source": pdf_path, "page": i + 1}))  # Store metadata

    if not documents:
        raise ValueError("Extracted text is empty. Please check the PDF content.")

    return documents

# Function to store resumes in FAISS index
def store_resumes_in_faiss(pdf_path, index_path=INDEX_DIRECTORY):
    documents = extract_text_pymupdf(pdf_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    if not text_chunks:
        raise ValueError("No valid text chunks found after splitting. Check the input text.")

    # Create FAISS vector store
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    
    # Save FAISS index locally
    vector_store.save_local(index_path)

# Background task to handle indexing
def background_indexing(pdf_path, index_path=INDEX_DIRECTORY):
    store_resumes_in_faiss(pdf_path, index_path)

Path(UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Upload PDF and start background indexing
@app.post("/upload-pdf/")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    # Save the uploaded file to the specified location
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Start background task to index the PDF
    background_tasks.add_task(background_indexing, file_location, INDEX_DIRECTORY)
    
    return {"message": f"PDF '{file.filename}' uploaded successfully!", "file_location": file_location}

# Function to load FAISS index
def load_faiss_index(index_path=INDEX_DIRECTORY):
    try:
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        raise FileNotFoundError("FAISS index not found. Store resumes first.") from e

# Define the request body model for the ask-candidate endpoint
class QuestionRequest(BaseModel):
    question: str

# Query endpoint to ask candidate questions
@app.post("/ask-candidate/")
async def ask_candidate_question(request: QuestionRequest):
    # Load FAISS index (make sure it's stored persistently)
    faiss_index = load_faiss_index(INDEX_DIRECTORY)
    
    # Set up retriever
    retriever = faiss_index.as_retriever(search_kwargs={"k": 3})
    
    # Define the prompt template for candidate Q&A
    template = """You are an AI assistant extracting candidate information from resumes.
    Given the retrieved resume content below, generate a detailed and structured answer.

    ### User Question:
    {query}

    ### Retrieved Resume Information:
    {context}

    ### Output:
    Provide a structured answer based on the resume details.
    """
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template=template
    )
    
    # Setup QA chain with Groq
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    
    # Get response
    response = qa_chain(request.question)
    
    # Format and return response
    return {
        "answer": response["result"],
        "source_documents": response["source_documents"]
    }

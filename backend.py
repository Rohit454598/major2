import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_mistralai import ChatMistralAI
from langchain.schema import HumanMessage, SystemMessage

# Initialize MistralAI model
llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key="MOzLnv20mK0lpN3SL48D0EmXdR7LSEuy")

# Initialize Sentence Transformer for embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper Function: Extract Text from PDF
def extract_text_from_pdf(file_path):
    """Extracts text from the given PDF file."""
    reader = PdfReader(file_path)
    text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

# Helper Function: Create FAISS Index
def create_faiss_index(texts, embedding_model):
    """Creates a FAISS index for the provided texts."""
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Helper Function: Retrieve Relevant Contexts
def retrieve_context(question, texts, index, embedding_model, top_k=3):
    """Retrieves the top-k relevant contexts for a given question."""
    question_embedding = embedding_model.encode([question], convert_to_tensor=False)
    distances, indices = index.search(np.array(question_embedding), top_k)
    relevant_texts = [texts[i] for i in indices[0]]
    return relevant_texts

# Process Lecture PDF
def process_lecture_pdf(pdf_path):
    """Processes a PDF, extracts text, and builds a FAISS index."""
    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    if not extracted_text:
        raise ValueError("No text could be extracted from the PDF.")

    # Split text into smaller chunks
    chunk_size = 300
    texts = [extracted_text[i:i + chunk_size] for i in range(0, len(extracted_text), chunk_size)]

    # Create a FAISS index
    index, embeddings = create_faiss_index(texts, embedding_model)
    return texts, index

# Answer Evaluation with RAG
def evaluate_answer_with_rag(question, student_answer, texts, index):
    """Evaluates an answer by retrieving relevant contexts and generating feedback."""
    # Retrieve relevant contexts
    relevant_contexts = retrieve_context(question, texts, index, embedding_model)
    combined_context = " ".join(relevant_contexts)

    # Define system prompt
    trait_definitions = (
        "You are an expert evaluator. Evaluate the student's answer based on the following traits:\n"
        "1. **Content**: Relevance and accuracy of the information with relevant context present in notes.\n"
        "2. **Coherence**: Logical flow and organization.\n"
        "3. **Vocabulary**: Range and appropriateness of vocabulary.\n"
        "4. **Grammar**: Correctness of language usage.\n\n"
        "explain what sentences need to change for betterment for each traits"
        "Provide detailed feedback on the answer's performance. At least 1000 words. Calculate overall score out of 10 give 50% weightage to content and 25% to Coherence."
    )

    system_message = SystemMessage(
        content=f"Use the following context to evaluate the student's answer:\n\n{combined_context}\n\n{trait_definitions}"
    )

    user_message = HumanMessage(content=f"Question: {question}\nAnswer: {student_answer}")

    # Use MistralAI to generate feedback
    response = llm([system_message, user_message])
    return response.content

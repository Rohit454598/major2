import streamlit as st
from backend import process_lecture_pdf, evaluate_answer_with_rag

st.title("Lecture Note Answer Evaluator")

# Upload PDF
uploaded_pdf = st.file_uploader("Upload your lecture notes (PDF):", type="pdf")

if uploaded_pdf is not None:
    with st.spinner("Processing PDF..."):
        # Save the uploaded file temporarily
        pdf_path = "uploaded_lecture_notes.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        # Process the PDF to extract texts and create FAISS index
        texts, faiss_index = process_lecture_pdf(pdf_path)

    st.success("PDF processed and FAISS index created.")

    # Input question and answer
    question = st.text_input("Enter a question:")
    student_answer = st.text_area("Enter the student's answer:")

    if st.button("Evaluate Answer"):
        if question and student_answer:
            with st.spinner("Evaluating answer..."):
                feedback = evaluate_answer_with_rag(question, student_answer, texts, faiss_index)
            st.subheader("Feedback:")
            st.write(feedback)
        else:
            st.error("Please provide both a question and an answer.")

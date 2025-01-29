import os
import streamlit as st
import fitz
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time


from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")


def get_pdf_processed(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    return text

def initialize_vector_store():
    pdf_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Submit & Process"):
        with st.spinner("Loading PDF..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            st.session_state.docs = get_pdf_processed(pdf_files)
            st.session_state.final_documents = st.session_state.text_splitter.split_text(st.session_state.docs)
            st.session_state.vector = FAISS.from_texts(st.session_state.final_documents, st.session_state.embeddings)
            st.success("PDF content loaded successfully!")

def get_conversational_chain():
    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the provided context only. 
        Please provide the most accurate response based on the question 
        <context> {context} </context> 
        Question: {input}"""
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector.as_retriever() if hasattr(st.session_state, 'vector') else None
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def user_input(prompt):
    chain = get_conversational_chain()
    start = time.process_time()
    response = chain.invoke({"input": prompt})
    st.write(response['answer'])
    st.write("Response time: ", time.process_time() - start)
    
    with st.expander("Expand to view your question in PDF"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------------")

def main():
    st.title("Ask Questions from PDF Documents")
    
    initialize_vector_store()
    
    prompt = st.text_input("Input your question here")
    if prompt:
        user_input(prompt)

if __name__ == "__main__":
    main()

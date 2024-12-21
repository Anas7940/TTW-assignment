import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_from_link(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except requests.RequestException as e:
        st.error(f"Failed to fetch text from the link: {url}. Error: {e}")
        return ""

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say "answer is not available in the context". 
    Do not provide a wrong answer.
    
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])

#Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF Documents and Links", layout="wide")
    st.header("Chat with PDF Documents and Links using Google Gemini 💁")

    user_question = st.text_input("Ask a Question from the Uploaded PDF or Link Content")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        links = st.text_area("Enter URLs (one per line):")

        if st.button("Submit & Process"):
            if pdf_docs or links.strip():
                with st.spinner("Processing..."):
                    # Extract text from the PDFs
                    raw_text = get_pdf_text(pdf_docs) if pdf_docs else ""

                    # Extract text from the links
                    if links.strip():
                        urls = links.strip().split("\n")
                        for url in urls:
                            raw_text += get_text_from_link(url)

                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)

                        get_vector_store(text_chunks)
                        st.success("Documents and links have been processed and indexed.")
                    else:
                        st.error("No valid text could be extracted from the provided inputs.")
            else:
                st.error("Please upload at least one PDF file or provide links.")

if __name__ == "__main__":
    main()

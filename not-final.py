import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from symbl import *
from symbl import *
from symbl import *
import fitz  
import io
import openai

openai.api_key = "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"

def is_legal_response(response):
    legal_keywords = ["law","sections", "sub-sections", "clause"]
    return any(keyword in response.lower() for keyword in legal_keywords)

def get_openai_response(question, model="gpt-3.5-turbo-0613"):
    messages = [
        {"role": "system", "content": "You are a helpful legal assistant specialized in Indian law."},
        {"role": "user", "content": f"Legal question: {question}"}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=1000
    )
    
    if is_legal_response(response['choices'][0]['message']['content']):
        return response['choices'][0]['message']['content'].strip()
    else:
        return "I don't know the answer to that question about Indian law."

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_file in pdf_docs:
        # Read the content of the BytesIO object
        pdf_content = io.BytesIO(pdf_file.read())

        # Create a document object
        doc = fitz.open(stream=pdf_content, filetype="pdf")

        for page_num in range(doc.page_count):
            # Get the page by index
            page = doc.load_page(page_num)

            # Read a Page
            text += page.get_text()

        doc.close()

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.1, max_tokens=2000)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, conversation_chain, pdf_processed):
    if conversation_chain and pdf_processed:
        response = conversation_chain({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
        
        # Check if the response is from the PDF, if not, fallback to GPT
        if pdf_processed and not response['chat_history']:
            response_text = response['chat_history'][0].content.lower() if response['chat_history'] else ""
            if is_legal_response(response_text):
                st.write(bot_template.replace("{{MSG}}", response['chat_history'][0].content), unsafe_allow_html=True)
            else:
                response = get_openai_response(user_question)
                st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
    else:
        response = get_openai_response(user_question)
        st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Legal Chatbot", page_icon="icon.png")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("HOW MAY I ASSIST YOU")
    user_question = st.text_input("ASK A LEGAL QUESTION RELATED TO INDIAN LAW")
    
    if user_question:
        handle_userinput(user_question, st.session_state.conversation, pdf_processed=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

if __name__ == '__main__':
    main()

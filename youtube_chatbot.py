import os
import re
import openai
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
# from langchain.document_loaders.generic import GenericLoader
# from langchain.document_loaders.parsers import OpenAIWhisperParser
# from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
# from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader

import PyPDF2
from pytube import YouTube

# Load environment variables
load_dotenv(find_dotenv())

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API = os.getenv("HUGGINGFACEHUB_API_KEY")

# LLM function
def generate_llm(all_splits):
    # We need a prompt that we can pass into an LLM to generate a transformed search query
    chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)
    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
            ),
        ]
    )

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    # k is the number of chunks to retrieve
    retriever = vectorstore.as_retriever(k=5)
    print("retriever is getting:", retriever.invoke("what is this video about?"))
    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            # If only one message, then we just pass that message's content to retriever
            (lambda x: x["messages"][-1].content) | retriever,
        ),
        # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
        query_transform_prompt | chat | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")
    
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

    # create retriever chains
    retrieval_chain = RunnablePassthrough.assign(
        context=query_transforming_retriever_chain,
    ).assign(
        answer=document_chain,
    )
    return retrieval_chain

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfFileReader(file)
    text = ''
    for page_num in range(reader.getNumPages()):
        text += reader.getPage(page_num).extract_text()
   
    metadata = reader.getDocumentInfo()
    title = metadata.title if metadata.title else 'No title available'
    
    return text, title, file.name

# Function to extract text from YouTube video
def extract_text_from_youtube(link):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", link)
    if match:
        video_id = match.group(1)
    else:
        raise ValueError("The provided link does not contain a valid YouTube video ID.")
    # transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    # docs = " ".join([entry['text'] for entry in transcript_list])
    loader = YoutubeLoader.from_youtube_url(
        link, add_video_info=False
    )   
    docs = loader.load()
    yt = YouTube(link)
    title = yt.title
    thumbnail_url = yt.thumbnail_url
    return docs, title, thumbnail_url

# Main Streamlit application
def main():
    st.set_page_config(page_title="YouTube/PDF Summarizer BOT", page_icon="🤖")
    st.header("Summarize and answer based on provided YouTube or PDF")
    
    # Initialize chat history, context, and necessary variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retrieval_chain" not in st.session_state:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_splits = text_splitter.create_documents("default")
        st.session_state.retrieval_chain = generate_llm(all_splits)
    if "chat_history" not in st.session_state:
        # Chat history for gpt
        st.session_state.chat_history = ChatMessageHistory()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    upload = False
    # File uploader for PDF in the sidebar
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            text, title, name = extract_text_from_pdf(uploaded_file)
            # Save text with metadata. This will automatically embed content
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            all_splits = text_splitter.create_documents(text)
            st.session_state.retriever = generate_llm(all_splits)
            upload = True
        # Display PDF title in chat message container
        with st.chat_message("assistant"):
            st.markdown(f"Successfully loaded the **Title:** {name}.pdf ")
        st.session_state.messages.append({"role": "assistant", "content": f"**Title:** {title}"})
            

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.add_user_message(prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Loading for YouTube link
        if "/youtube" in prompt:
            with st.spinner("Processing YouTube video..."):
                text, title, thumbnail_url = extract_text_from_youtube(prompt[8:]) #index 8 = after "/youtube"
                # Save text with metadata. This will automatically embed content
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                all_splits = text_splitter.split_documents(text)
                st.session_state.retriever = generate_llm(all_splits)
                upload = True
            # Display YouTube video thumbnail and title in chat message container
            with st.chat_message("assistant"):
                st.markdown("Successfully loaded provided YouTube video! Feel free to ask anything about it.")
                st.markdown(f"**Title:** {title}")
                st.image(thumbnail_url, width=200)  # Adjusted the width for smaller size
        
            st.session_state.messages.append({"role": "assistant", 
                                                "content": f"Successfully loaded provided YouTube video! Feel free to ask anything about it. **Title:** {title}\n"
                                                f"<img src='{thumbnail_url}' width='200' alt='Thumbnail'>"})
            st.session_state.chat_history.add_ai_message(f"Successfully loaded provided YouTube video! Feel free to ask anything about it. **Title:** {title}\n")

        if not upload:
            # Invoking the retriever above results in some parts that chatbot can use as context when answering questions.
            # Get the response from the retriever and document chain
            result = st.session_state.retrieval_chain.invoke({"messages": st.session_state.chat_history.messages})
            answer = result["answer"]
            context = result["context"]

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.markdown(f"<details><summary>Source Context</summary><pre>{context}</pre></details>", unsafe_allow_html=True)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.messages.append({"role": "assistant", "content": f"<details><summary>Source Context</summary><pre>{context}</pre></details>"})
            st.session_state.chat_history.add_ai_message(answer)

if __name__ == "__main__":
    main()

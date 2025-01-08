import os
import cohere
import pinecone
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.llms import Cohere as LangCohere
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from PyPDF2 import PdfReader
from docx import Document
import streamlit as st

# Load environment variables from a .env file
load_dotenv()

# Initialize Cohere API client
cohere_client = cohere.Client(os.getenv('YOUR_COHERE_API_KEY'))

# Initialize Pinecone client
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv('YOUR_PINECONE_API_KEY'))

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []
if 'file_content' not in st.session_state:
    st.session_state['file_content'] = ""

# Setting page title and header
st.set_page_config(page_title="Chat GPT Clone", page_icon=":robot_face:") 
st.markdown("<h1 style='text-align: center;'>Chat with Your Files📄🤖</h1>", unsafe_allow_html=True)

# Initialize Pinecone index
index_name = "file-contents"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=4096, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(index_name)

# Sidebar for file uploads and summarization
uploaded_files = st.sidebar.file_uploader("Upload Files (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
summarise_button = st.sidebar.button("Summarise Conversation")

# Function to process uploaded files 
def process_uploaded_files(files):
    content = ""
    for file in files:
        file_type = file.name.split('.')[-1].lower()
        if file_type == 'pdf':
            reader = PdfReader(file)
            for page in reader.pages:
                content += page.extract_text()
        elif file_type == 'docx':
            doc = Document(file)
            content += "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_type == 'txt':
            content += file.read().decode("utf-8")
        else:
            st.error(f"Unsupported file type: {file_type}")
    return content

# Function to generate embedding using Cohere
def generate_embeddings(text):
    response = cohere_client.embed(texts=[text], model="embed-english-v2.0", truncate="END")
    embedding = response.embeddings[0]
    return embedding

# Function to store file content in Pinecone
def store_in_pinecone(content, file_name):
    # Generate embedding for file content
    embedding = generate_embeddings(content)
    
    # Store the embedding in Pinecone
    metadata = {"file_name": file_name, "content": content[:1000000]} 
    index.upsert([(file_name, embedding, metadata)])
 
# Process uploaded files
if uploaded_files:
    file_content = process_uploaded_files(uploaded_files)
    st.session_state['file_content'] = file_content
    st.sidebar.success("Files uploaded and processed successfully!")

    # Store the content in Pinecone
    for file in uploaded_files:
        store_in_pinecone(file_content, file.name)

# Summarize button functionality
if summarise_button and st.session_state['conversation']:
    st.sidebar.write("Summary of conversation:\n\n" + st.session_state['conversation'].memory.buffer)

# Function to handle chatbot responses
def getresponse(userInput):
    if st.session_state['conversation'] is None:
        llm = LangCohere(temperature=0, model="command-xlarge-nightly")
        st.session_state['conversation'] = ConversationChain(
            llm=llm,
            verbose=True,
            memory=ConversationSummaryMemory(llm=llm)
        )

    # Add file content as context
    context = st.session_state['file_content']
    if context:
        userInput = f"Context: {context}\nQuestion: {userInput}"

    response = st.session_state['conversation'].predict(input=userInput)
    return response


# Function to fetch all stored file data from Pinecone
def fetch_all_from_pinecone():
    try:
        # Perform a query with a dummy vector and large top_k to fetch all data
        query_response = index.query(
            vector=[0] * 4096,  # Dummy vector of appropriate dimensions
            top_k=100,  # Increase this if you expect more records
            include_metadata=True
        )
        fetched_content = ""
        for match in query_response.matches:
            fetched_content += match.metadata.get("content", "")
        return fetched_content
    except Exception as e:
        st.error(f"Error fetching all data from Pinecone: {e}")
        return ""

# Restore session state on refresh
if not st.session_state.get('file_content'):
    # Attempt to fetch all stored data from Pinecone
    restored_content = fetch_all_from_pinecone()
    if restored_content:
        st.session_state['file_content'] = restored_content
        st.sidebar.success("Restored file content from Pinecone.")
    else:
        st.sidebar.warning("No data found in Pinecone to restore.")

# Process uploaded files
if uploaded_files:
    file_content = process_uploaded_files(uploaded_files)
    st.session_state['file_content'] = file_content
    st.sidebar.success("Files uploaded and processed successfully!")

    # Store the content in Pinecone
    try:
        for file in uploaded_files:
            store_in_pinecone(file_content, file.name)
    except Exception as e:
        st.error(f"Error storing data in Pinecone: {e}")



# Chat UI
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your question goes here:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        if submit_button:
            st.session_state['messages'].append(user_input)
            model_response = getresponse(user_input)
            st.session_state['messages'].append(model_response)

            with response_container:
                for i in range(len(st.session_state['messages'])):
                    if (i % 2) == 0:
                        message(st.session_state['messages'][i], is_user=True, key=str(i) + '_user')
                    else:
                        message(st.session_state['messages'][i], key=str(i) + '_AI')

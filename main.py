import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
import tempfile

# Load environment variables
load_dotenv()

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def init_pinecone():
    """Initialize Pinecone client with error handling"""
    try:
        from pinecone import Pinecone
        pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        )
        return pc
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        return None

def process_pdf(uploaded_file):
    """Process uploaded PDF file and create embeddings"""
    try:
        # Create a temporary directory to store the PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_path = os.path.join(temp_dir, "uploaded_document.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load and process the PDF
            loader = PyPDFDirectoryLoader(temp_dir)
            documents = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50
            )
            doc_chunks = text_splitter.split_documents(documents)
            return doc_chunks
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def setup_qa_system(doc_chunks):
    """Set up the question-answering system with embeddings and vector store"""
    try:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize Pinecone and create index
        pc = init_pinecone()
        if pc is None:
            return None
        
        index_name = os.getenv("PINECONE_INDEX_NAME", "langchainindex")
        
        # Create or get existing index
        vector_store = Pinecone.from_documents(
            doc_chunks,
            embeddings,
            index_name=index_name
        )
        
        # Initialize QA chain
        llm = OpenAI(
            model_name="gpt-3.5-turbo-instruct",
            temperature=0.5,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            verbose=True
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error setting up QA system: {str(e)}")
        return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI PDF Question Answering",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("AI PDF Question Answering System")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        st.markdown("Upload your PDF document and ask questions about its content.")
        
        # API key input fields
        openai_key = st.text_input("OpenAI API Key:", type="password")
        pinecone_key = st.text_input("Pinecone API Key:", type="password")
        
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if pinecone_key:
            os.environ["PINECONE_API_KEY"] = pinecone_key

    # Main content area
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            doc_chunks = process_pdf(uploaded_file)
            
            if doc_chunks:
                qa_chain = setup_qa_system(doc_chunks)
                
                if qa_chain:
                    # Query input
                    query = st.text_input("Ask a question about the document:")
                    
                    if query:
                        with st.spinner("Searching for answer..."):
                            try:
                                # Get answer from QA chain
                                result = qa_chain({
                                    "question": query,
                                    "chat_history": st.session_state.conversation_history
                                })
                                
                                # Display answer
                                st.markdown("### Answer:")
                                st.write(result["answer"])
                                
                                # Update conversation history
                                st.session_state.conversation_history.append(
                                    (query, result["answer"])
                                )
                                
                                # Display conversation history
                                if st.session_state.conversation_history:
                                    st.markdown("### Conversation History:")
                                    for q, a in st.session_state.conversation_history:
                                        st.markdown(f"**Q:** {q}")
                                        st.markdown(f"**A:** {a}")
                                        st.markdown("---")
                            
                            except Exception as e:
                                st.error(f"Error getting answer: {str(e)}")

if __name__ == "__main__":
    main()
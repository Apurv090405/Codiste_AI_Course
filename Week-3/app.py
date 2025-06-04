import streamlit as st
import os
from dotenv import load_dotenv
import hnswlib
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import PyPDF2
from unstructured.partition.auto import partition

# Load .env
load_dotenv()

# Get Google API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env file. Please set it and try again.")
    st.stop()

# Set the API key for Gemini
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Initialize Gemini embeddings and LLM
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="RETRIEVAL_DOCUMENT")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3) # temp = 0.3 for more deterministic not more random and creative 

# Step 1: Generate embeddings
@st.cache_data
def generate_embeddings(docs):
    try:
        embeddings = embeddings_model.embed_documents(docs)
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

# Step 2: Initialize HNSW (Hierarchical Navigable Small World graphs) index and similarity search
# HNSW uses graph-based indexing for efficient nearest neighbor search.
# Compared to FAISS, HNSW uses a graph-based approach for approximate nearest neighbor search, while FAISS supports multiple indexing methods (e.g., IVF).
# HNSW is often preferred for small to medium datasets due to its efficient graph-based search, while FAISS may be faster for large datasets with certain indexing methods.        
def initialize_hnsw_index(embeddings, docs, metas, dimension):
    try:
        index = hnswlib.Index(space="cosine", dim=dimension)  # Uses cosine distance for similarity search, suitable for high-dimensional text embeddings in HNSW.
        # Other distance metrics like Euclidean (L2) are supported but less common for text embeddings, which are often normalized for cosine distance.
        index.init_index(max_elements=len(embeddings) + 10, ef_construction=100, M=16)  
        # max_elements: Maximum number of items the index can store.
        # ef_construction: Size of the dynamic candidate list during graph construction, affecting build time and index quality.
        # M: Number of bi-directional links for each element.
        index.add_items(embeddings, list(range(len(embeddings))))
        index.set_ef(50)
        # ef: Size of the dynamic candidate list during similarity search, balancing speed and accuracy.
        return index, docs, metas
    except Exception as e:
        st.error(f"Error initializing HNSW index: {e}")
        return None, docs, metas
    

# Step 3: Similarity search
def similarity_search(index, docs, metas, query_embedding, k=2):
    try:
        num_elements = index.get_current_count()
        k = min(k, num_elements) if num_elements > 0 else 0
        if k == 0:
            st.warning("No documents in the index.")
            return []
        labels, _ = index.knn_query(query_embedding, k=k)
        return [Document(page_content=docs[i], metadata=metas[i]) for i in labels[0]]
    except Exception as e:
        st.error(f"Error in similarity search: {e}")
        return []

# Step 4: RAG implementation
def create_rag_chain(index, docs, metas):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following context to answer the question:\n{context}"),
        ("user", "{question}")
    ])
    
    def retrieve_and_generate(query):
        try:
            query_embedding = embeddings_model.embed_query(query)
            retrieved_docs = similarity_search(index, docs, metas, query_embedding, k=2)
            if not retrieved_docs:
                return "No relevant documents found.", []
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            messages = prompt_template.invoke({"question": query, "context": context})
            response = llm.invoke(messages)
            return response.content, retrieved_docs
        except Exception as e:
            st.error(f"Error in RAG chain: {e}")
            return "An error occurred while processing the query.", []

    return retrieve_and_generate

# Function to extract text from uploaded files (PDF or TXT)
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        elif uploaded_file.name.endswith(".txt"):
            elements = partition(file=uploaded_file)
            text = "\n".join([str(el) for el in elements])
            return text
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {e}")
        return None

# Streamlit app
st.title("RAG Chatbot with Gemini and HNSW")
st.write("Upload PDF or TXT Files to create a knowlage base, then ask a question.")

upload_file = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
if upload_file:
    documents = []
    metadata = []
    for i, file in enumerate(upload_file):
        text = extract_text_from_file(file)
        if text:
            documents.append(text)
            metadata.append({"id": i, "source": file.name})
    if documents:
        st.success(f"Successfully extracted {len(documents)} documents.")
        # Generate and store embeddings
        embeddings = generate_embeddings(documents)
        if embeddings is not None:
            dimension = len(embeddings[0])  # Typically 768 for embedding-001
            hnsw_index, stored_docs, stored_metadata = initialize_hnsw_index(embeddings, documents, metadata, dimension)
            
            if hnsw_index is not None:
                # Create RAG chain
                rag_chain = create_rag_chain(hnsw_index, stored_docs, stored_metadata)
                
                # User input
                query = st.text_input("Enter your question:", "What is machine learning?")
                if st.button("Submit"):
                    if query:
                        with st.spinner("Generating answer..."):
                            answer, retrieved_docs = rag_chain(query)
                            st.write("**Answer**:")
                            st.write(answer)
                            st.write("**Retrieved Documents**:")
                            for i, doc in enumerate(retrieved_docs, 1):
                                st.write(f"{i}")
                    else:
                        st.write("Please enter a question.")
            else:
                st.error("Failed to initialize HNSW index.")
        else:
            st.error("Failed to generate embeddings.")
    else:
        st.error("Documents could not be extracted from the uploaded files.")
else:
    st.write("Please upload at least one PDF or TXT file to get started.")
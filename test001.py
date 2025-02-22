import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1️⃣ Load Hugging Face Token Securely
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Load from environment variable

if not HUGGINGFACE_TOKEN:
    raise ValueError("Please set HUGGINGFACE_TOKEN in your environment variables.")

# 2️⃣ Load IBM Granite 3B Model and Tokenizer
model_name = "ibm/granite-3b-moe"  # Ensure this model exists and you have access

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",            # Auto-detect GPU/CPU
    low_cpu_mem_usage=True,
    use_auth_token=HUGGINGFACE_TOKEN
)

# 3️⃣ Create Hugging Face Pipeline for LLM
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# 4️⃣ Load Embedding Model for Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5️⃣ Load Knowledge Base from Text File
loader = TextLoader("text.txt")  # Ensure this file exists!
documents = loader.load()

# 6️⃣ Split Text into Chunks for RAG Processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 7️⃣ Create FAISS Vector Store
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 8️⃣ Set Up RAG Retrieval Pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_pipeline,
    retriever=vectorstore.as_retriever()
)

# 9️⃣ Ask a Question and Retrieve Context
query = "What is IBM Granite 3.1 used for?"
response = qa_chain.run(query)
print(response)
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# Load existing vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# LLM via Groq
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("API_KEY"),
    temperature=0.2
)

# Prompt
prompt = PromptTemplate.from_template("""
You are a helpful assistant with deep knowledge of The Lord of the Rings.
Use only the context below to answer. If you don't know, say so.

Context:
{context}

Question: {question}
Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Modern LCEL chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat loop
print("LOTR RAG Chatbot — type 'quit' to exit\n")
while True:
    q = input("You: ")
    if q.lower() == "quit":
        break
    answer = chain.invoke(q)
    print(f"Bot: {answer}\n")
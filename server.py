#!/usr/bin/env python3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import os

# Load environment variables
model = os.environ.get("MODEL", "phi4")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

# Load constants
from constants import CHROMA_SETTINGS

# Initialize FastAPI app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
llm = Ollama(model=model)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# In-memory history store
chat_histories = {}

class QueryRequest(BaseModel):
    session_id: str
    question: str
    hide_source: bool = False

@app.post("/ask")
async def ask_question(request: QueryRequest):
    session_id = request.session_id
    question = request.question.strip()

    if not question:
        return {"error": "Empty question"}

    chat_history = chat_histories.get(session_id, [])

    # Step 1: Retrieve documents
    docs = retriever.get_relevant_documents(question)

    # Step 2: Basic relevance check — no documents returned
    if not docs:
        return {
            "question": question,
            "answer": "Sorry, I cannot answer that. It seems to be out of context.",
            "sources": [],
            "chat_history": chat_history,
        }

    # Step 3: Optionally apply a content-based relevance check (optional, see Option 2 below)

    # Step 4: Run the chain as usual
    custom_prompt = f"Answer the following in a clear, well-formatted way:\n\n{question}"
    result = qa_chain.invoke({"question": custom_prompt, "chat_history": chat_history})
    answer = result["answer"]
    docs = result.get("source_documents", [])
    chat_history.append((question, answer))
    chat_histories[session_id] = chat_history

    sources = []
    if not request.hide_source:
        sources = [
            {"source": doc.metadata.get("source", "Unknown"), "content": doc.page_content}
            for doc in docs
        ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "chat_history": chat_history,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

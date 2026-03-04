import os
from dotenv import load_dotenv
from google import genai

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------
# Load Vectorstore Once
# ---------------------------
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# ---------------------------
# Load Gemini Client
# ---------------------------
def load_gemini():

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    return client


# ---------------------------
# Generate Answer
# ---------------------------
def generate_answer(query):

    vectorstore = load_vectorstore()
    client = load_gemini()

    # Get similarity scores
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=5)

    # FAISS: lower score = more similar
    SIMILARITY_THRESHOLD = 0.6  # tune if needed

    relevant_docs = [
        doc for doc, score in docs_and_scores if score < SIMILARITY_THRESHOLD
    ]

    # HARD GUARD: if no relevant chunks, skip LLM entirely
    if not relevant_docs:
        return "I could not find this information in the Swiggy Annual Report.", []

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are a financial document assistant.

Answer strictly using ONLY the provided context.

If the answer is not explicitly present in the context,
respond exactly with:

"I could not find this information in the Swiggy Annual Report."

Do NOT use outside knowledge.
Do NOT infer.
Do NOT guess.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )

    return response.text.strip(), relevant_docs


# ---------------------------
# Main Loop
# ---------------------------
if __name__ == "__main__":

    print("\n📄 Swiggy Annual Report RAG \n")

    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer, docs = generate_answer(query)

        print("\nANSWER:\n")
        print(answer)

        if docs:
            print("\n--- Supporting Pages ---")
            for doc in docs:
                print("Page:", doc.metadata.get("page_label"))
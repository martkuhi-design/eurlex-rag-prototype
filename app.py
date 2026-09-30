import streamlit as st
import json
import math
from openai import OpenAI

# Load stored article embeddings
with open("article_embeddings.json", "r", encoding="utf-8") as f:
    article_embeddings = json.load(f)

# Connect to OpenAI
client = OpenAI()

st.write("Loaded articles:", len(article_embeddings))

st.title("EU Electronic Communications Code – AI Assistant")

st.write(
    "Ask a question about Directive (EU) 2018/1972 "
    "(European Electronic Communications Code)."
)

question = st.text_input("Your question:")

if question:
    # Turn the user's question into an embedding
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    query_embedding = query_response.data[0].embedding

    # Compare two embeddings
    def cosine_similarity(a, b):
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(y * y for y in b))

        return dot_product / (magnitude_a * magnitude_b)

    # Compare the question with all 127 articles
    semantic_results = []

    for item in article_embeddings:
        similarity = cosine_similarity(
            query_embedding,
            item["embedding"]
        )

        semantic_results.append({
            "article": item["article"],
            "similarity": similarity,
            "text": item["text"]
        })

    # Highest similarity first
    semantic_results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # Show the three most relevant articles
    st.write("Most relevant articles:")

    for result in semantic_results[:3]:
        st.write(
            f"Article {result['article']} | "
            f"similarity: {result['similarity']:.4f}"
        )

        # Take the 3 most relevant articles
        top_results = semantic_results[:3]

        # Combine their texts into one context for the language model
        context = "\n\n".join(
            f"ARTICLE {item['article']}\n{item['text']}"
            for item in top_results
        )

        st.subheader("Retrieved context")

        with st.expander("Show retrieved article text"):
            st.text(context)

        # Ask the language model to answer using only the retrieved context
        prompt = f"""
        You are an assistant answering questions about Directive (EU) 2018/1972.

        Answer the user's question using only the context provided below.

        Rules:
        - Do not use information outside the provided context.
        - Cite the relevant Article number(s) in your answer.
        - If the context does not contain enough information to answer the question, say so clearly.
        - Give a concise and clear answer.

        USER QUESTION:
        {question}

        CONTEXT:
        {context}
        """

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        answer = response.output_text

        st.subheader("Answer")
        st.write(answer)

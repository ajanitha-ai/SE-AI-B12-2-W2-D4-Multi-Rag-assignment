
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

st.title("Multi-PDF RAG Assistant")

st.write(
    "Upload multiple PDF documents and ask questions "
    "about their contents."
)

uploaded_files = st.file_uploader(
    "Choose your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"You uploaded {len(uploaded_files)} PDF file(s).")

    for pdf in uploaded_files:
        st.write(f"📄 {pdf.name}")

    if st.button("Read PDF contents"):
        all_pages = []

        for pdf in uploaded_files:
            # Save the uploaded PDF temporarily so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as temp_file:
                temp_file.write(pdf.getvalue())
                temp_path = temp_file.name

            try:
                loader = PyPDFLoader(temp_path)
                pages = loader.load()

                # Keep the original uploaded filename in the source metadata
                for page in pages:
                    page.metadata["source"] = pdf.name

                all_pages.extend(pages)

            finally:
                os.remove(temp_path)

        st.session_state["pdf_pages"] = all_pages
        all_chunks = text_splitter.split_documents(all_pages)

        st.session_state["pdf_chunks"] = all_chunks

        store = FAISS.from_documents(all_chunks, embeddings)
        st.session_state["store"] = store

        st.success(f"Indexed {len(all_chunks)} chunks into the vector database.")


        st.success(f"Created {len(all_chunks)} text chunks.")

        for i, chunk in enumerate(all_chunks[:5]):
            st.write(f"**Chunk {i + 1}**")
            st.write(f"Source: {chunk.metadata['source']}")
            st.write(chunk.page_content)
            st.divider()

        st.success(
            f"Successfully read {len(all_pages)} page(s) "
            f"from {len(uploaded_files)} PDF file(s)."
        )

        for page in all_pages:
            st.write(
                f"**File:** {page.metadata['source']} "
                f"| **Page:** {page.metadata['page'] + 1}"
            )
            st.write(page.page_content[:300] or "(No text extracted)")
            st.divider()


    if "store" in st.session_state:
        st.subheader("Ask questions about your PDFs")

        question = st.text_input(
            "Enter your question:"
        )

        ask_button = st.button("Ask Question")
        
        if ask_button and question:
            retriever = st.session_state["store"].as_retriever(
                search_kwargs={"k": 4}
            )
            relevant_docs = retriever.invoke(question)
            
            context = "\n\n".join(
                f"Source: {doc.metadata['source']}, "
                f"Page: {doc.metadata['page'] + 1}\n"
                f"{doc.page_content}"
                for doc in relevant_docs
            )

            prompt = f"""
You are a helpful project management assistant.
Answer the user's question using only the provided document context.

Rules:
- Do not invent facts that are not in the context.
- If the documents contain conflicting information, explain the conflict.
- Distinguish proposed dates from formally approved dates.
- If the context does not contain the answer, say so.
- Cite the source filename and page number for factual claims.

Document context:
{context}

User question:
{question}
"""

            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response.content)
            st.subheader("Sources used")
            for doc in relevant_docs:
                st.write(
                    f"- {doc.metadata['source']} "
                    f"(Page {doc.metadata['page'] + 1})"
                )



            st.subheader("Relevant passages found")

            for doc in relevant_docs:
                st.write(f"**File:** {doc.metadata['source']}")
                st.write(f"**Page:** {doc.metadata['page'] + 1}")
                st.write(doc.page_content)
                st.divider()


else:
    st.info("Please upload one or more PDF documents to begin.")


# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq
# # # from langchain.chains.question_answering import load_qa_chain
# # # from langchain_community.chains.question_answering import load_qa_chain
# # from langchain_community.chains.question_answering import load_qa_chain
# # import os

# # os.environ["GROQ_API_KEY"] = "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"


# # #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
# # st.header("My RAG BOT")

# # # Upload the PDF
# # with st.sidebar:
# #     st.title("Your Document")
# #     file = st.file_uploader("Upload your PDF", type="pdf")

# # # Extract the text
# # if file is not None:
# #     pdf_pages = PdfReader(file)
# #     text = ""

# #     # Extract text from each page
# #     for page in pdf_pages.pages:
# #         text += page.extract_text() or ""

# #     st.subheader("Extracted Text")
# #     ##st.write(text)

# #     text_splitter = RecursiveCharacterTextSplitter(
# #         separators="\n",
# #         chunk_size = 1000,
# #         chunk_overlap = 150,
# #         length_function = len
# #     )
# #     chunks = text_splitter.split_text(text)

# #     model_name = "sentence-transformers/all-mpnet-base-v2"
# #     # model_kwargs = {"device": "cpu"}
# #     # encode_kwargs = {"normalize_embeddings": False}
# #     embeds = HuggingFaceEmbeddings(
# #         model_name=model_name,
# #     )

# #     vector_store =  FAISS.from_texts(chunks,embeds)

# #     user_query = st.text_input("Enter the query")

# #     if user_query:
# #         match = vector_store.similarity_search(user_query)
# #         llm = ChatGroq(
# #             model="llama-3.1-8b-instant",
# #             temperature=0.0,
# #             max_retries=2,
# #         )

# #         #chain = load_qa_chain(llm, chain_type = "stuff")

# #         #response = chain.run(input_documents = match, question = user_query)
# #         respose = ""

# #         st.subheader("ANSWER")
# #         st.write(response)


# # else:
# #     st.write("Please upload a PDF to extract its text.")


# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq
# import os

# # Set Groq API key
# os.environ["GROQ_API_KEY"] = "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"

# st.header("My RAG BOT")

# # Sidebar upload
# with st.sidebar:
#     st.title("Your Document")
#     file = st.file_uploader("Upload your PDF", type="pdf")

# # Process PDF
# if file is not None:
#     pdf = PdfReader(file)
#     text = ""

#     # Extract text from pages
#     for page in pdf.pages:
#         extracted = page.extract_text()
#         if extracted:
#             text += extracted

#     # Stop if no text found
#     if len(text.strip()) == 0:
#         st.error("❌ No readable text found — PDF might be scanned. OCR needed.")
#         st.stop()

#     # Split text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n"],
#         chunk_size=1000,
#         chunk_overlap=150,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)

#     if len(chunks) == 0:
#         st.error("❌ Failed to create chunks — PDF may be image-based.")
#         st.stop()

#     # Embeddings
#     model_name = "sentence-transformers/all-mpnet-base-v2"
#     embeddings = HuggingFaceEmbeddings(model_name=model_name)

#     # Vectorstore
#     vector_store = FAISS.from_texts(chunks, embeddings)

#     # User query
#     user_query = st.text_input("Enter your query")

#     if user_query:
#         # Get top chunks
#         matched_chunks = vector_store.similarity_search(user_query, k=4)

#         # Build context
#         context = "\n\n".join([c.page_content if hasattr(c, 'page_content') else str(c) for c in matched_chunks])

#         # Final prompt to Groq LLM
#         prompt = f"""
# You are an AI assistant. Use ONLY the context below to answer the question.

# Context:
# {context}

# Question: {user_query}

# Answer in simple, clear language:
# """

#         llm = ChatGroq(
#             model="llama-3.1-8b-instant",
#             temperature=0.0,
#             max_retries=2,
#         )

#         # LLM call
#         response = llm.invoke(prompt)

#         st.subheader("ANSWER")
#         st.write(response.content)

# else:
#     st.write("Please upload a PDF to extract its text.")


from streamlit_float import float_init
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os

os.environ["GROQ_API_KEY"] = "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"

st.set_page_config(page_title="RAG Bot + Chatbot", layout="wide")
st.header("📄 My RAG PDF + Floating Chatbot")

# ------------------------
# SIDEBAR: PDF UPLOAD + CHATBOX
# ------------------------
with st.sidebar:
    st.title("📚 Upload Your Document")
    file = st.file_uploader("Upload PDF", type="pdf")
    st.write("---")
    st.subheader("🤖 Floating Chatbot (Separate Model)")
    user_chat_msg = st.text_input("Message the chatbot")

# ------------------------
# PDF -> RAG pipeline
# ------------------------
if file is not None:
    pdf = PdfReader(file)
    text = ""

    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if len(text.strip()) == 0:
        st.error("❌ No readable text found — PDF might be scanned. OCR needed.")
        st.stop()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    if len(chunks) == 0:
        st.error("❌ Could not create text chunks.")
        st.stop()

    # Embeddings & vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Query input
    user_query = st.text_input("🔍 Ask something about your PDF")

    if user_query:
        matched_chunks = vector_store.similarity_search(user_query, k=4)

        # Build context text
        context = "\n\n".join([c.page_content if hasattr(c, "page_content") else str(c) for c in matched_chunks])

        prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question: {user_query}

Answer in clear, concise language:
"""

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        rag_resp = llm.invoke(prompt)

        st.subheader("📌 Answer (RAG)")
        # some SDK responses return object with .content; handle both
        if hasattr(rag_resp, "content"):
            st.write(rag_resp.content)
        else:
            st.write(str(rag_resp))

else:
    st.info("📄 Upload a PDF to begin.")

# ------------------------
# FLOATING BUTTON (center-right) - CSS injected via st.markdown
# ------------------------
st.markdown(
    """
    <style>
    #chatbot-btn {
        position: fixed;
        bottom: 45%;
        right: 20px;
        background: #4F46E5;
        color: white;
        border-radius: 40px;
        padding: 14px;
        font-size: 28px;
        cursor: pointer;
        transition: transform 0.15s ease;
        z-index: 9999;
        box-shadow: 0 6px 18px rgba(79,70,229,0.25);
    }
    #chatbot-btn:hover { transform: scale(1.08); background: #4338CA; }
    </style>

    <div id="chatbot-btn" title="Open chatbot">🤖</div>
    """,
    unsafe_allow_html=True,
)

# Note: the floating button is purely visual here. The chat input/output are in the sidebar (above).
# You can wire JavaScript to open a modal, but Streamlit's ability to run custom JS is limited without extra hacks.

# ------------------------
# SIDEBAR CHATBOT LOGIC (uses a different model)
# ------------------------
if user_chat_msg:
    chatbot_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)  # different model
    chat_resp = chatbot_llm.invoke(user_chat_msg)

    st.sidebar.write("### 💬 Bot Response")
    if hasattr(chat_resp, "content"):
        st.sidebar.write(chat_resp.content)
    else:
        st.sidebar.write(str(chat_resp))
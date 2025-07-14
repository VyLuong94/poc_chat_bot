
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
import os

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="Chatbot Thu Hồi Nợ", layout="wide")
st.markdown("<h1 style='text-align: center;'>🤖 Chatbot Thu Hồi Nợ</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_chain():
    # Tích hợp SOP nội bộ trực tiếp trong code
    sop_text = """
    Q: Nếu khách hàng xin giãn nợ do khó khăn tài chính, cần xử lý thế nào?
    A: Hướng dẫn khách gửi đơn đề nghị giãn nợ kèm giấy tờ chứng minh. Chuyển hồ sơ về bộ phận hỗ trợ đặc biệt.

    Q: Khách hàng phản ứng tiêu cực và đòi kiện, phải làm gì?
    A: Ghi nhận vào CRM, ngưng gọi cuộc đó và chuyển hồ sơ sang bộ phận pháp lý.

    Q: Khách hàng muốn trả góp lại khoản nợ quá hạn?
    A: Không đồng ý trực tiếp. Ghi nhận thiện chí và gửi đề xuất về bộ phận tái cấu trúc hợp đồng.
    """

    # Tạo Document từ văn bản trực tiếp
    documents = [Document(page_content=sop_text)]
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=retriever)
    return chain

qa_chain = load_chain()

# Giao diện chat giống ChatGPT
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.container():
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"🧑 **Bạn:** {message}")
        else:
            st.markdown(f"🤖 **Chatbot:** {message}")

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.chat_history.append(("user", query))
    with st.spinner("🤔 Đang tìm câu trả lời..."):
        answer = qa_chain.run(query)
    st.session_state.chat_history.append(("bot", answer))
    st.rerun()

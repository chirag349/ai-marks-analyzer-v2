import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Marks Analyzer",
    layout="centered"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#141E30);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
h1,h2,h3,h4,h5,h6,p,label { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("AI-Assisted Answer Evaluation System")
st.subheader("Teacher 👨🏻‍🏫")

col1, col2 = st.columns(2)
with col1:
    st.text_area("Paste Reference Answer", key="ref_text")
with col2:
    st.text_area("Paste Question Paper (optional)", key="ques_text")

st.subheader("Student 🧑‍🎓")
st.text_area("Paste Student Answer", key="stu_text")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def calculate_similarity(reference_text, student_text):
    documents = [reference_text, student_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return score

if st.button("Evaluate"):
    reference_text = clean_text(st.session_state.ref_text)
    student_text = clean_text(st.session_state.stu_text)

    if reference_text.strip() == "" or student_text.strip() == "":
        st.error("Please paste both reference answer and student answer.")
    else:
        similarity_score = calculate_similarity(reference_text, student_text)

        max_marks = 10
        if similarity_score >= 0.95:
            final_marks = max_marks
        else:
            final_marks = round(similarity_score * max_marks, 2)

        st.subheader("Evaluation Result")
        st.write("Similarity Score:", round(similarity_score, 3))
        st.write("Final Marks:", final_marks, "/", max_marks)

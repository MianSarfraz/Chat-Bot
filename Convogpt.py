import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import csv
from datetime import datetime
import random

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("bpp_university_qa.csv")
        df['context'] = df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        return df
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None

# Search function using TF-IDF and Cosine Similarity
def semantic_search(query, df):
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = [query] + df['context'].tolist()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    df['similarity'] = cosine_similarities
    most_relevant_row = df.loc[df['similarity'].idxmax()]
    return most_relevant_row if most_relevant_row['similarity'] > 0 else None

# Function to log user questions and bot answers
def log_interaction(question, answer):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "user_interactions.csv"
    
    try:
        with open(log_file, 'x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["Timestamp", "Question", "Answer"])
    except FileExistsError:
        pass
    
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([timestamp, question, answer])

# Basic communication responses
def basic_communication(query):
    query = query.lower()
    responses = {
        "greetings": ["Hello! How can I assist you today?", "Hi there! What would you like to know?", "Greetings! How may I help you?"],
        "farewells": ["Goodbye! Have a great day!", "Farewell! Feel free to come back if you have more questions.", "See you later! Take care!"],
        "thanks": ["You're welcome!", "Happy to help!", "My pleasure!"]
    }
    
    for category, phrases in [("greetings", ["hello", "hi", "hey", "greetings"]),
                              ("farewells", ["bye", "goodbye", "see you", "farewell"]),
                              ("thanks", ["thank you", "thanks", "appreciate it"])]:
        if any(word in query for word in phrases):
            return random.choice(responses[category])
    
    return None

# Custom CSS for enhanced white and blue theme
def set_custom_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        color: #333333;
    }
    .stApp {
        background-color: #F0F8FF;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #007BFF;
        border-color: #007BFF;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,123,255,0.2);
    }
    .stTextInput>div>div>input {
        color: #333333;
        background-color: #FFFFFF;
        border-radius: 25px;
        border: 2px solid #007BFF;
        padding: 12px 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #007BFF;
        font-weight: 700;
    }
    .stDataFrame {
        border: 1px solid #007BFF;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,123,255,0.1);
    }
    .stProgress > div > div > div {
        background-color: #007BFF;
    }
    .stSpinner > div > div {
        border-top-color: #007BFF;
    }
    .stAlert {
        background-color: #E6F3FF;
        color: #007BFF;
        border-radius: 15px;
        border: 1px solid #007BFF;
        padding: 16px;
        margin-bottom: 1rem;
    }
    .stExpander {
        border: 1px solid #007BFF;
        border-radius: 15px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .stExpander > div > div > div > div {
        background-color: #FFFFFF;
        color: #333333;
        padding: 16px;
    }
    .stTabs > div > div > div {
        background-color: #FFFFFF;
        border-radius: 15px;
        border: 1px solid #007BFF;
        padding: 16px;
        margin-top: 1rem;
    }
    .stTabs > div > div > div > div > div {
        color: #333333;
    }
    .css-1544g2n {
        padding: 2rem 1rem;
    }
    .css-1544g2n h1 {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to process the user's question and generate a response
def process_question(question, data):
    progress_bar = st.progress(0)
    
    # Check for basic communication first
    basic_response = basic_communication(question)
    if basic_response:
        st.markdown(f"<p style='color: #333333; background-color: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #007BFF;'>{basic_response}</p>", unsafe_allow_html=True)
        log_interaction(question, basic_response)
        return

    # Perform semantic search
    with st.spinner("Searching for relevant information..."):
        relevant_data = semantic_search(question, data)
        progress_bar.progress(50)

    # Display result
    tab1, tab2 = st.tabs(["Answer", "Context"])
    
    with tab1:
        st.markdown(f"<h3 style='color: #007BFF;'>Question</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #333333; background-color: #E6F3FF; padding: 10px; border-radius: 10px;'>{question}</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: #007BFF;'>Answer</h3>", unsafe_allow_html=True)
        
        if relevant_data is not None:
            answer = relevant_data['context']
            st.markdown(f"<p style='color: #333333; background-color: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #007BFF;'>{answer}</p>", unsafe_allow_html=True)
            log_interaction(question, answer)
        else:
            answer = "Sorry, no relevant information found in our database for your question."
            st.markdown(f"<p style='color: #333333; background-color: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #007BFF;'>{answer}</p>", unsafe_allow_html=True)
            log_interaction(question, answer)
    
    with tab2:
        if relevant_data is not None:
            st.subheader("Most Relevant Context")
            st.markdown(f"<p style='color: #333333;'>{relevant_data['context']}</p>", unsafe_allow_html=True)

    progress_bar.progress(100)
    time.sleep(0.5)
    progress_bar.empty()

# Main app
def main():
    st.set_page_config(page_title="ConvoGpt - BPP University QA", layout="wide")
    set_custom_theme()

    # Sidebar
    with st.sidebar:
        st.image("https://convosoft.com/wp-content/uploads/2023/05/Convosoft-Logo-png-white_CNVO-02-1024x1024.png", width=100)
        st.title("ConvoGpt - BPP University QA")
        st.markdown("---")
        st.subheader("About")
        st.write("This AI-powered assistant provides answers to your queries related to BPP University.")
        st.markdown("---")
        st.write("## Instructions")
        st.write("Type your question in the text box below and click 'Ask'.")

    # Load data
    df = load_and_preprocess_data()
    if df is not None:
        st.header("Ask Your Question")
        user_input = st.text_input("Your question:")
        
        if st.button("Ask"):
            if user_input:
                process_question(user_input, df)
            else:
                st.warning("Please enter a question.")

# Run the app
if __name__ == "__main__":
    main()

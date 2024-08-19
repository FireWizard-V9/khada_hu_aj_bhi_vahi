import streamlit as st 
import rag_backend as demo  ### replace rag_backend with your backend filename
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Read In Between Reports",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background: url("https://path-to-your-background-image.jpg") no-repeat center center fixed;
            background-size: cover;
            padding: 20px;
        }
        .title {
            font-family: 'Arial', sans-serif;
            color: #006400;
            font-size: 42px;
            text-align: center;
        }
        .input-area {
            font-size: 16px;
        }
        .button {
            background-color: #006400;
            color: white;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 18px;
        }
        .button:hover {
            background-color: #004d00;
        }
        .footer {
            text-align: center;
            padding: 10px;
            margin-top: 20px;
            color: #888;
        }
        .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 150px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">INFOSYS HR QNA 🎯</p>', unsafe_allow_html=True)

# Session state for vector index
if 'vector_index' not in st.session_state: 
    with st.spinner("📀 Wait for magic... All beautiful things in life take time :-)"):  ### Spinner message
        st.session_state.vector_index = demo.hr_index()  ### Your Index Function name from Backend File

# Layout using columns
col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area("Input text", label_visibility="collapsed", placeholder="Type your question here...", height=200, help="Enter the text or question you want to search for.") 

with col2:
    st.markdown('<br><br>', unsafe_allow_html=True)  # Add some space
    go_button = st.button("📌 Do your Search", type="primary", use_container_width=True, key="search_button", help="Click to search for the best match.")  ### Button Name

# Display response
if go_button: 
    with st.spinner("📢 I will read in between lines while you drink your coffee..."):  ### Spinner message
        response_content = demo.hr_rag_response(index=st.session_state.vector_index, question=input_text)  ### Replace with RAG Function from backend file
        st.write(response_content)

# Footer
st.markdown("""
    <div class="footer">
        Developed with ❤️ by Sahil<br>
        All rights reserved.
    </div>
""", unsafe_allow_html=True)

import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu

from gemini_utility import (
    load_gemini_pro_model,
    gemini_pro_response,
    gemini_pro_vision_response,
    embeddings_model_response
)

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Chat Bot",
    page_icon="🚀",
    layout="centered",
)
st.markdown("<h1 style='text-align: center;'>Institute of Technology of Cambodia</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Department of AMS</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>🤖 Play With Chat Bot 📊</h2>", unsafe_allow_html=True)
# Custom CSS for the navigation bar
st.markdown("""
    <style>
        .nav-container {
            display: flex;
            justify-content: center;
            background-color: #2E4053;
            padding: 10px;
        }
        .nav-item {
            margin: 0 15px;
            color: white;
            font-size: 18px;
            cursor: pointer;
        }
        .nav-item:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for selected option if not already present
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = 'ChatBot'

# Horizontal navigation bar
def set_selected_option(option):
    st.session_state.selected_option = option

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("ChatBot"):
            set_selected_option("ChatBot")
    with col2:
        if st.button("Image Captioning"):
            set_selected_option("Image Captioning")
    with col3:
        if st.button("Embed Text"):
            set_selected_option("Embed Text")
    with col4:
        if st.button("Ask me anything"):
            set_selected_option("Ask me anything")

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

# ChatBot page
if st.session_state.selected_option == 'ChatBot':
    model = load_gemini_pro_model()

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chatbot's title on the page


    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask me anything...")
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

# Image Captioning page
if st.session_state.selected_option == "Image Captioning":
    st.title("📷 Snap Narrate")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption"):
        if uploaded_image is not None:
            image = Image.open(uploaded_image)

            col1, col2 = st.columns(2)

            with col1:
                resized_img = image.resize((800, 500))
                st.image(resized_img)

            default_prompt = "write a short caption for this image"

            # Get the caption of the image from the gemini_pro_vision LLM
            caption = gemini_pro_vision_response(default_prompt, image)

            with col2:
                st.info(caption)
        else:
            st.warning("Please upload an image first.")

# Text Embedding page
if st.session_state.selected_option == "Embed Text":
    st.title("🔡 Embed Text")

    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

    if st.button("Get Response"):
        if user_prompt:
            response = embeddings_model_response(user_prompt)
            st.markdown(response)
        else:
            st.warning("Please enter some text to get embeddings.")

# Ask Me Anything page
if st.session_state.selected_option == "Ask me anything":
    st.title("❔ Ask me a question")

    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response"):
        if user_prompt:
            response = gemini_pro_response(user_prompt)
            st.markdown(response)
        else:
            st.warning("Please enter a question to get a response.")

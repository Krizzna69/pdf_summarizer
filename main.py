import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import textwrap
import time

# Set your Google Generative AI API Key here
API_KEY = 'AIzaSyBQab0Iu4QieYyiQLbYHwFjlGJLkQTd6us'
# Configure Google Generative AI
def configure_gemini_api_key(key):
    genai.configure(api_key=key)

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle None case
    return text

# Function to summarize text using Google Gemini
def summarize_text_with_gemini(text_batch):
    generation_config = {
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }

    llm = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
    )

    try:
        response = llm.generate_content(f"Summarize the following text:\n{text_batch}")
        return response.text
    except Exception as e:
        st.error(f"Error: {e}")
        return ""

# Split the text into manageable batches
def split_text_into_batches(text, max_length=4000):
    return textwrap.wrap(text, width=max_length, break_long_words=False)

# Stream lit Interface
def main():
    st.markdown(
        """
        <style>
        .title {
            font-size: 40px;
            color: #4CAF50;
            text-align: center;
        }
        .header {
            font-size: 30px;
            color: #333;
        }
        .text-area {
            border: 2px solid #4CAF50;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("AI-Based Virtual Library")
    st.write("Upload a PDF, and we'll extract and summarize its content in batches using Google Gemini API.")

    # Configure the API key
    configure_gemini_api_key(API_KEY)

    # File upload
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Extracting text from the PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

        if not extracted_text:
            st.error("No text could be extracted from the PDF.")
            return

        st.success("Text extracted successfully!")

        # Display extracted text (optional)
        if st.checkbox("Show Extracted Text"):
            st.text_area("Extracted Text", extracted_text, height=300)

        # Summarize text
        st.write("### Summarization in Progress")
        batches = split_text_into_batches(extracted_text)
        summary = ""

        for i, batch in enumerate(batches):
            with st.spinner(f"Summarizing batch {i+1}/{len(batches)}..."):
                summary_part = summarize_text_with_gemini(batch)
                summary += summary_part + "\n\n"
                time.sleep(1)  # Optional: Add a delay to avoid hitting API rate limits

        st.success("Summarization completed!")

        # Display summary
        st.write("### Summary")
        st.text_area("Summary", summary, height=300)

        # Option to download the summary
        st.download_button("Download Summary", summary, file_name="summary.txt")

if __name__ == "__main__":
    main()
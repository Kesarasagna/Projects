import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle
import os
import re
import pdfplumber
from PyPDF2 import PdfReader
import docx

model = joblib.load('random_forest_model.pkl')

with open('random_forest_model.pkl', 'rb') as f:
    tfidf = pickle.load(f)
model = joblib.load('naive_bayes_model.pkl')

with open('naive_bayes_model.pkl', 'rb') as f:
    nb = pickle.load(f)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_category(resume_text):
    Cleaned_Resume = clean_text(resume_text)

category_map = {
    0: "PeopleSoft",
    1: "React Developer",
    3: "SQL Developer",
    4: "workday"
}

def categorize_resumes(uploaded_files, output_directory):
    if not os.path.exists(output_directory):
      os.makedirs(output_directory)

      results = []

      for uploaded_file in uploaded_files:
          if uploaded_file.name.endswith('.txt','.pdf','.docx','DOCX'):
            reader = PdfReader(uploaded_file)
            page = reader.pages[0]
            text = page.extract_text()
            Cleaned_Resume = clean_text(text)

            input_features = model.transform([Cleaned_Resume])
            prediction_id = model.predict(input_features)[0]
            category_id = category_map.get(prediction_id[0])

            category_folder = os.path.join(output_directory, category_id)

            if not os.path.exists(category_folder):
                os.makedirs(category_folder)

            target_file = os.path.join(category_folder, uploaded_file.name)
            with open(target_file, 'w') as f:
                f.write(uploaded_file.read().decode('utf-8'))

            results.append({'File Name': uploaded_file.name, 'Category': category_id})

      results_df = pd.DataFrame(results)

      return results_df

def main():

    st.title("Resume Categorizer Application")
    st.subheader("With Python & Machine Learning")

    uploaded_files = st.file_uploader("Upload your resumes", type=["pdf"], accept_multiple_files=True)
    uploaded_files = st.file_uploader("Upload your resumes", type=["docx"], accept_multiple_files=True)
    uploaded_files = st.file_uploader("Upload your resumes", type=["doc"], accept_multiple_files=True)

    output_directory = st.text_input("Output Directory", "categorized_resumes")

    if st.button("Categorize_Resumes"):
        if uploaded_files and output_directory:
            results_df = categorize_resumes(uploaded_files, output_directory)
            st.write(results_df)
            results_csv = results_df.to_csv(index=False).encode('utf-8')

            st.download_button(
            label="Download results as CSV",
            data=results_csv,
            file_name='Cleaned_Resumes.csv',
            mime='text/csv',
        )

        st.success("Resumes categorization and processing completed.")
    else:
        st.error("Please upload files and specify the output directory.")

if __name__ == '__main__':
    main()



import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader  # Optional for handling PDFs

# Load the trained model
with open('naive_bayes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    # Remove non-alphabet characters, convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("Resume Classification App")
st.write("Upload a resume and classify it into one of the categories!")

uploaded_file = st.file_uploader("Upload your resume (docx or PDF)", type=["docx", "pdf", "doc"])

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Update file uploader handling
if uploaded_file is not None:
    try:
        content = uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
        content = uploaded_file.read().decode('windows-1252', errors='ignore')

    # Display uploaded content
    st.subheader("Uploaded Resume Content:")
    st.write(content)

    # Preprocess the text
    preprocessed_text = preprocess_text(content)

    # Transform the text using the vectorizer
    vectorized_text = vectorizer.transform([preprocessed_text])

    # Predict the category
    prediction = model.predict(vectorized_text)[0]

    category_mapping = {
    0: "PeopleSoft",
    1: "React Developer",
    2: "SQL Developer",
    3: "workday"
    }

    # Convert prediction to label
    predicted_category = category_mapping[prediction]
    st.write(f"**Category:** {predicted_category}")

    # Display the result
    st.subheader("Predicted Resume Category:")
    st.write(f"**{prediction}**")
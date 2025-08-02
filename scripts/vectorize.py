import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # To save vectorizers and matrices

def load_documents(folder_path):
    docs = []
    filenames = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
                filenames.append(filename)
    return docs, filenames

def vectorize_documents(docs, save_path=None):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)
    if save_path:
        joblib.dump(vectorizer, save_path)
    return vectors, vectorizer

if __name__ == "__main__":
    # Load cleaned resumes and job descriptions
    resume_docs, resume_files = load_documents("data/resume_cleaned")
    jd_docs, jd_files = load_documents("data/jd_cleaned")

    # Combine for global TF-IDF (alternatively, vectorize separately)
    all_docs = resume_docs + jd_docs
    tfidf_matrix, vectorizer = vectorize_documents(all_docs, "outputs/tfidf_vectorizer.pkl")

    # Split back to resume and JD vectors
    resume_vectors = tfidf_matrix[:len(resume_docs)]
    jd_vectors = tfidf_matrix[len(resume_docs):]

    # Save individual parts for later use
    joblib.dump(resume_vectors, "outputs/resume_vectors.pkl")
    joblib.dump(jd_vectors, "outputs/jd_vectors.pkl")
    pd.Series(resume_files).to_csv("outputs/resume_filenames.csv", index=False)
    pd.Series(jd_files).to_csv("outputs/jd_filenames.csv", index=False)

    print("TF-IDF vectorization completed.")

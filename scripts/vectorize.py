import os
import joblib
import spacy
import numpy as np

def load_documents(folder_path):
    docs = []
    filenames = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
                filenames.append(filename)
    return docs, filenames

def get_doc_vector(nlp, text):
    doc = nlp(text)
    if doc.vector_norm:
        return doc.vector
    else:
        return np.zeros(nlp.vocab.vectors_length)

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_md")

    resumes, resume_files = load_documents("data/resume_cleaned")
    jds, jd_files = load_documents("data/jd_cleaned")

    resume_vectors = [get_doc_vector(nlp, text) for text in resumes]
    jd_vectors = [get_doc_vector(nlp, text) for text in jds]

    joblib.dump(resume_vectors, "outputs/resume_vectors.pkl")
    joblib.dump(jd_vectors, "outputs/jd_vectors.pkl")
    joblib.dump(resume_files, "outputs/resume_files.pkl")
    joblib.dump(jd_files, "outputs/jd_files.pkl")

    print("✔ Saved spaCy-based document vectors.")

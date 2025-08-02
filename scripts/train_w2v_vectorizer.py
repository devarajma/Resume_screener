import os
import joblib
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np

# Download required NLTK resources
nltk.download('stopwords')

# Use regex tokenizer to avoid punkt_tab error
tokenizer = RegexpTokenizer(r'\w+')

def load_docs(folder):
    texts = []
    filenames = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
                filenames.append(fname)
    return texts, filenames

def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    stops = set(stopwords.words("english"))
    return [word for word in tokens if word.isalpha() and word not in stops]

def average_vector(tokens, model, size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(size)

if __name__ == "__main__":
    # Load and clean resumes
    resumes, resume_files = load_docs("data/resume_cleaned")
    jds, jd_files = load_docs("data/jd_cleaned")

    # Combine and tokenize for training
    all_texts = resumes + jds
    tokenized = [preprocess(text) for text in all_texts]

    print("🔄 Training Word2Vec model...")
    w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    w2v_model.save("outputs/word2vec_model.model")

    print("✅ Generating average vectors for resumes and job descriptions...")
    resume_vectors = [average_vector(preprocess(text), w2v_model, 100) for text in resumes]
    jd_vectors = [average_vector(preprocess(text), w2v_model, 100) for text in jds]

    # Save vectors and filenames
    joblib.dump(resume_vectors, "outputs/resume_vectors_w2v.pkl")
    joblib.dump(jd_vectors, "outputs/jd_vectors_w2v.pkl")
    joblib.dump(resume_files, "outputs/resume_files.pkl")
    joblib.dump(jd_files, "outputs/jd_files.pkl")

    print("🎉 Word2Vec vectors saved successfully.")

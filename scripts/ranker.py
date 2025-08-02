import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    resume_vectors = joblib.load("outputs/resume_vectors.pkl")
    jd_vectors = joblib.load("outputs/jd_vectors.pkl")
    resume_files = joblib.load("outputs/resume_files.pkl")
    jd_files = joblib.load("outputs/jd_files.pkl")
    resume_vecs = joblib.load("outputs/resume_vectors_w2v.pkl")
    jd_vecs = joblib.load("outputs/jd_vectors_w2v.pkl")

    return resume_vectors, jd_vectors, resume_files, jd_files

def rank_resumes(resume_vectors, jd_vector, resume_files, top_n=5):
    similarities = cosine_similarity([jd_vector], resume_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1][:top_n]
    return [(resume_files[i], similarities[i]) for i in ranked_indices]

if __name__ == "__main__":
    resume_vecs, jd_vecs, resume_files, jd_files = load_data()

    for i, jd_vector in enumerate(jd_vecs):
        print(f"\n📄 Job Description: {jd_files[i]}")
        top_matches = rank_resumes(resume_vecs, jd_vector, resume_files, top_n=5)
        for rank, (filename, score) in enumerate(top_matches, 1):
            print(f"{rank}. {filename} (Similarity: {score:.4f})")

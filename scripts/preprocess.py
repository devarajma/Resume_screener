import os
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_digit and token.is_alpha
    ]
    return " ".join(tokens)

def preprocess_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            path = os.path.join(input_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned_text = preprocess_text(raw_text)

            out_path = os.path.join(output_dir, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"Preprocessed: {filename}")

# Example usage
if __name__ == "__main__":
    preprocess_folder("data/resume_texts", "data/resume_cleaned")
    preprocess_folder("data/job_descriptions", "data/jd_cleaned")

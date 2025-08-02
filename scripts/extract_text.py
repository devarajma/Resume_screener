import os
import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def extract_from_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            txt_filename = filename.replace(".pdf", ".txt")
            with open(os.path.join(output_dir, txt_filename), "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Extracted: {filename} -> {txt_filename}")

# Example usage
if __name__ == "__main__":
    extract_from_folder("data/resumes", "data/resume_texts")
    extract_from_folder("data/job_descriptions", "data/jd_texts")

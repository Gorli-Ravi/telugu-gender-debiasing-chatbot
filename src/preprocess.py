import re

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r"[^\u0C00-\u0C7F\s]", "", text)  # Keep only Telugu characters
    return text.strip()

def preprocess_dataset(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        if line.strip():
            cleaned_lines.append(clean_text(line))
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))

if __name__ == "__main__":
    input_path = "../data/telugu_gender_bias_dataset.txt"
    output_path = "../data/cleaned_dataset.txt"
    preprocess_dataset(input_path, output_path)
    print("✅ Preprocessing complete! Cleaned dataset saved to", output_path)


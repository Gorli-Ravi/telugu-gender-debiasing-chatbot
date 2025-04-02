import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import subprocess

# Ensure required dependencies are installed
def install_dependencies():
    try:
        import tiktoken
    except ImportError:
        subprocess.run(["pip", "install", "tiktoken"], check=True)
    try:
        import google.protobuf
    except ImportError:
        subprocess.run(["pip", "install", "protobuf"], check=True)
    try:
        import blobfile
    except ImportError:
        subprocess.run(["pip", "install", "blobfile"], check=True)
    try:
        import sentencepiece
    except ImportError:
        subprocess.run(["pip", "install", "sentencepiece"], check=True)

install_dependencies()

def load_model():
    model_name = "google/mt5-small"  # Using Google mT5-small model for Telugu
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # Force using SentencePiece
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

def predict(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def test_with_dataset(model, tokenizer, dataset_path, device):
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for line in lines[:5]:  # Test first 5 sentences
        if line.startswith("Biased:"):
            biased_text = line.replace("Biased:", "").strip()
            neutral_text = predict(model, tokenizer, biased_text, device)
            print(f"Original: {biased_text}\nNeutral: {neutral_text}\n")

if __name__ == "__main__":
    model, tokenizer, device = load_model()
    dataset_path = "../data/telugu_gender_bias_dataset.txt"
    test_with_dataset(model, tokenizer, dataset_path, device)


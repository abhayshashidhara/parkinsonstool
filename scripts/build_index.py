import json, os, faiss, torch, numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import timm
from torchvision import transforms
from PIL import Image

# Load JSONL datasets
path_1 = "/kaggle/input/first-code-data/NEWparkinsons_synthetic_20k (1).jsonl"
path_2 = "/kaggle/input/first-code-data/parkinsons_diagnostic_cases (1).jsonl"

with open(path_1, "r", encoding="utf-8") as f1:
    dataset_1 = [json.loads(line) for line in f1]
with open(path_2, "r", encoding="utf-8") as f2:
    dataset_2 = [json.loads(line) for line in f2]

merged_dataset = dataset_1 + dataset_2

# Build case chunks
case_chunks = []
for entry in merged_dataset:
    case_chunks.append(
        "[CASE]\n" + entry["input"].strip() + "\n\n" + entry["output"].strip()
    )

# Read PDFs to chunks
pdf_folder = "/kaggle/input/paper-data"
pdf_chunks = []
for fname in os.listdir(pdf_folder):
    if not fname.lower().endswith(".pdf"):
        continue
    reader = PdfReader(os.path.join(pdf_folder, fname))
    full_text = ""
    for page in reader.pages:
        txt = page.extract_text() or ""
        full_text += txt + "\n\n"
    for i in range(0, len(full_text), 2000):
        chunk = full_text[i:i+2000].strip()
        if len(chunk) > 200:
            pdf_chunks.append(f"[PAPER:{fname}]\n" + chunk)

# Embed + index
all_chunks = case_chunks + pdf_chunks
embedder = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")
embeddings = embedder.encode(all_chunks, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "faiss_index.index")
print("FAISS index saved as faiss_index.index")
print(f"Loaded {len(case_chunks)} cases + {len(pdf_chunks)} papers = {len(all_chunks)} total.")

# Load LLM
model_id = "openchat/openchat-3.5-0106"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
rag_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

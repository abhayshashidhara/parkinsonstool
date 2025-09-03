
import faiss, torch, numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PIL import Image
from torchvision import transforms
import timm

# Load the FAISS index againn
index = faiss.read_index("faiss_index.index")

# Embedder for the user query
embedder = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

# Language model for drafting the write-up
model_id = "openchat/openchat-3.5-0106"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
generate = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Spiral classifier: preprocessing and weights
device = "cuda" if torch.cuda.is_available() else "cpu"

spiral_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

spiral_model = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=2)
state = torch.load("/kaggle/working/spiral_model.pth", map_location=device)
spiral_model.load_state_dict(state)
spiral_model.eval().to(device)

def predict_spiral_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    x = spiral_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = spiral_model(x)
        pred = int(torch.argmax(out, dim=1).item())
    return ("The spiral test suggests signs of Parkinson’s."
            if pred == 1 else
            "The spiral test does not show signs of Parkinson’s.")

def diagnose(patient_text: str, image_path: str | None = None, k: int = 3) -> str:
    # Optional spiral note
    spiral_note = "No spiral test image was provided."
    if image_path:
        spiral_note = predict_spiral_image(image_path)

    # Retrieve nearest chunks 
    q = embedder.encode([patient_text], convert_to_numpy=True)
    _dists, _idxs = index.search(q, k)

    prompt = f"""You are a senior neurologist. Based on the patient's clinical information, generate a medically realistic diagnostic summary using the format below. Include the spiral analysis result.

1. Patient Overview
2. Symptom Summary
3. Clinical Impression
4. Reasoning
5. Spiral Test Image Analysis
{spiral_note}

Patient Details:
{patient_text.strip()}

#END OF PROMPT ANSWER BELOW #
"""
    out = generate(prompt, max_new_tokens=400, do_sample=False, temperature=0.7)[0]["generated_text"]
    marker = "# END OF PROMPT — ANSWER BELOW #"
    return out.split(marker, 1)[1].strip() if marker in out else out.strip()

if __name__ == "__main__":
    patient = """
    Age: 60
    Gender: Male
    Smoking: No
    Alcohol: Yes
    Diet: Non-Vegetarian
    Allergies: Pollen
    Comorbidities: None
    Surgery: Brain surgery
    Immunosuppressants: No
    Genetic History: Father has Parkinson's Disease
    Symptoms:
    - Bradykinesia Score: 2.2
    - Tremors: Yes
    - Rigidity: Mild
    - Gait Disturbance: Yes
    - Speech Changes: No
    - Sleep Disturbance: Yes
    - Balance Issues: Yes
    - Cognitive: No
    """
    spiral_img = "/kaggle/input/kaggledatafortesting/parkinson/V01PE01.png"
    print(diagnose(patient, image_path=spiral_img))

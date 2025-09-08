!pip install -q gradio timm

import gradio as gr
import timm
import torch
from torchvision import transforms
from PIL import Image

model = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=2)
model.load_state_dict(torch.load("/kaggle/working/spiral_model.pth", map_location=torch.device("cpu")))
model.eval()

spiral_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def analyze_spiral(image):
    img = Image.open(image).convert("RGB")
    img_tensor = spiral_transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img_tensor)
        pred = torch.argmax(out, dim=1).item()
    return ("The spiral test suggests signs of Parkinson’s."
            if pred == 1 else
            "The spiral test does not show signs of Parkinson’s.")

demo = gr.Interface(
    fn=analyze_spiral,
    inputs=gr.Image(type="filepath", label="Upload Spiral Test Image"),
    outputs=gr.Textbox(label="Spiral Test Result"),
    title="Parkinson's Spiral Test Analyzer",
    description="Upload a spiral drawing image to analyze possible signs of Parkinson’s disease.",
    allow_flagging="never"
)

def rag_ui(age, gender, smoking, alcohol, diet, allergies, comorbidities, surgery, immuno, genetic_history,
           bradykinesia, tremors, rigidity, gait, speech, sleep, constipation, balance, extra_notes, spiral_image):
    user_input = f"""
Age: {age}
Gender: {gender}
Smoking: {smoking}
Alcohol: {alcohol}
Diet: {diet}
Allergies: {allergies}
Comorbidities: {comorbidities}
Surgery: {surgery}
Immunosuppressants: {immuno}
Genetic History: {genetic_history}
Symptoms:
- Bradykinesia Score: {bradykinesia}
- Tremors: {tremors}
- Rigidity: {rigidity}
- Gait Disturbance: {gait}
- Speech Changes: {speech}
- Sleep Disturbance: {sleep}
- Constipation: {constipation}
- Balance Issues: {balance}
Extra Notes: {extra_notes}
""".strip()

    spiral_image_path = spiral_image if spiral_image else None
    diagnosis = rag_chat_mistral(user_input, image_path=spiral_image_path)
    return diagnosis

ui = gr.Interface(
    fn=rag_ui,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio(["No", "Occasionally", "Daily"], label="Smoking"),
        gr.Radio(["No", "Occasionally", "Daily"], label="Alcohol"),
        gr.Dropdown(["Vegetarian", "Non-Vegetarian", "Mixed"], label="Diet"),
        gr.Textbox(label="Allergies", placeholder="None"),
        gr.Textbox(label="Comorbidities", placeholder="e.g. Diabetes, Hypertension"),
        gr.Textbox(label="Surgery History", placeholder="e.g. Bypass surgery, None"),
        gr.Radio(["Yes", "No"], label="Immunosuppressants"),
        gr.Textbox(label="Genetic History", placeholder="e.g. Father had PD, None"),
        gr.Slider(minimum=0, maximum=5, step=0.1, label="Bradykinesia Score"),
        gr.Radio(["Yes", "No"], label="Tremors"),
        gr.Radio(["Yes", "No"], label="Rigidity"),
        gr.Radio(["Yes", "No"], label="Gait Disturbance"),
        gr.Radio(["Yes", "No"], label="Speech Changes"),
        gr.Radio(["Yes", "No"], label="Sleep Disturbance"),
        gr.Radio(["Yes", "No"], label="Constipation"),
        gr.Radio(["Yes", "No"], label="Balance Issues"),
        gr.Textbox(label="Extra Notes", placeholder="Any extra symptoms or lifestyle notes"),
        gr.Image(label="Upload Spiral Test Image (Optional)", type="filepath"),
    ],
    outputs=gr.Textbox(label="Diagnosis"),
    title="Parkinson's Diagnosis Assistant"
)

ui.launch(debug=True, share=True)

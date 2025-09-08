# Integrated diagnostic: optional spiral check + retrieval + LLM
def generate_diagnosis(patient_text, image_path=None, k=3):
    # Spiral note (only if an image is provided)
    spiral_note = "No spiral test image was provided."
    if image_path:
        spiral_note = predict_spiral_image(image_path)

    # Retrieve nearest chunks (not injected into the prompt here)
    query = embedder.encode([patient_text], convert_to_numpy=True)
    _dists, _idxs = index.search(query, k)

    # Prompt for the LLM
    prompt = f"""You are a senior neurologist. Based on the patient's clinical information, generate a medically realistic diagnostic summary using the format below. Include the spiral analysis result.

1. Patient Overview
Summarise the patient’s demographics and background in one paragraph, touching on:
- Age and gender
- Smoking and alcohol habits
- Diet
- Recent surgeries or ongoing medications
- Family history of neurological disorders

2. Symptom Summary
Describe each of these in its own sentence:
- Bradykinesia Score
- Tremors (present/absent)
- Rigidity (none/mild/moderate/severe)
- Gait disturbance
- Speech changes
- Sleep disturbances
- Balance issues

3. Clinical Impression
- “Definitive Parkinson’s disease (likely)”
- “Probable Parkinson’s (likely)”
- “Possible Parkinsonism—consider alternative etiologies”
- “Unlikely Parkinson’s disease”

4. Reasoning
2–3 sentences linking findings to impression.

5. Spiral Test Image Analysis
{spiral_note}



Patient Details:
{patient_text.strip()}



"""

    # Generate
    out = rag_pipe(prompt, max_new_tokens=400, do_sample=False, temperature=0.7)[0]["generated_text"]

    # Return text after the marker
    marker = "### END OF PROMPT — ANSWER BELOW ###"
    return out.split(marker, 1)[1].strip() if marker in out else out.strip()


# Example just to check 
test_prompt = """
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

print("Doctor's Diagnosis:\n")
print(generate_diagnosis(test_prompt))

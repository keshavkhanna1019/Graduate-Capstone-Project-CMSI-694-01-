# Assignment #2 Supplemental Guide
## Example Projects & Quick-Start Code

---

## Example Project Ideas (Pre-Vetted for Scope)

### Tier 1: Quickest to Complete (2–2.5 hours)

#### 1. Hugging Face Sentiment Analysis
**What it does:** User enters text → returns sentiment (positive/negative/neutral) + confidence

**Pre-trained model:** `distilbert-base-uncased-finetuned-sst-2-english`

**Minimal code example (Streamlit):**
```python
import streamlit as st
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="distilbert-base-uncased-finetuned-sst-2-english")

st.title("Sentiment Analyzer")
user_text = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    result = sentiment_pipeline(user_text)
    st.write(f"Sentiment: {result[0]['label']}")
    st.write(f"Confidence: {result[0]['score']:.2%}")
```

**Prep time:** ~10 minutes setup, ~30 minutes adding error handling and better UI

---

#### 2. ImageNet Classification (PyTorch)
**What it does:** User uploads image → returns predicted object + confidence

**Pre-trained model:** `resnet50` from `torchvision.models`

**Minimal code example (Flask):**
```python
from flask import Flask, request, jsonify
from torchvision import models, transforms
from PIL import Image
import torch

app = Flask(__name__)
model = models.resnet50(pretrained=True)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    return jsonify({"prediction": predicted.item()})

if __name__ == '__main__':
    app.run(debug=True)
```

**Prep time:** ~15 minutes setup, ~40 minutes for UI and error handling

---

#### 3. Zero-Shot Classification
**What it does:** User enters text and candidate labels → returns best match

**Pre-trained model:** `facebook/bart-large-mnli` (zero-shot classifier from Hugging Face)

**Why it's quick:** No training needed; just classification API

**Prep time:** ~20 minutes total

---

### Tier 2: Moderate Complexity (3–3.5 hours)

#### 4. Fine-Tuned Text Classification (Custom Domain)
**What it does:** Classify news headlines into categories (sports, tech, politics, etc.)

**Approach:**
- Use a small labeled dataset (Kaggle has many free options)
- Fine-tune `distilbert-base-uncased` using Hugging Face Trainer (1–2 epochs)
- Wrap in a web app

**Libraries:**
- Hugging Face Transformers + Datasets
- Flask or Streamlit for UI

**Estimated time:**
- Data prep: ~30 minutes
- Fine-tuning: ~45 minutes (including hyperparameter search)
- Web integration: ~1 hour

---

#### 5. Time Series Forecasting
**What it does:** User enters historical data (stock prices, temperature) → predicts next values

**Pre-trained or custom LSTM/GRU model**

**Data source:** yfinance, OpenWeather API, or provided CSV

**Minimal code example:**
```python
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions
```

**Estimated time:** 2.5–3.5 hours (including data handling)

---

#### 6. Audio Event Detection
**What it does:** User uploads audio → detects what sound is in it (dog bark, traffic, speech, etc.)

**Pre-trained model:** 
- `openai/whisper` (speech-to-text + classification)
- Or `PANNs` (pretrained audio neural networks)

**Prep time:** ~2.5–3 hours

---

### Tier 3: More Ambitious (4+ hours, good for bonus credit)

#### 7. Image Generation (Fast Diffusion Model)
**What it does:** User enters text prompt → generates image

**Pre-trained model:** `stabilityai/stable-diffusion-2` (requires API key) or `runwayml/stable-diffusion-v1-5` (open-source)

**Warning:** Model is large (~5GB); inference is slow on CPU (30+ seconds)

**Best approach:** Use inference-optimized libraries like `diffusers`

**Prep time:** 3–4 hours

---

#### 8. Multi-Modal: Image + Text → Caption
**What it does:** User uploads image → generates descriptive caption

**Pre-trained model:** `microsoft/git-base` or `nlpconnect/vit-gpt2-image-captioning`

**Estimated time:** 2.5–3.5 hours

---

#### 9. Named Entity Recognition (NER)
**What it does:** User enters text → highlights people, places, organizations, etc.

**Pre-trained model:** `dbmdz/bert-base-german-cased` or `dslim/bert-base-NER`

**Estimated time:** 2–2.5 hours

---

## Tech Stack Quick-Start Templates

### Template 1: Streamlit (Fastest Prototyping)

```bash
# Install
pip install streamlit torch transformers pillow

# Example app (save as app.py)
```

```python
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="ML App", layout="wide")
st.title("My Deep Learning Model")

# Load model once (cached)
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

model = load_model()

# UI
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file)
    
    if st.button("Classify"):
        try:
            with st.spinner("Analyzing..."):
                results = model(uploaded_file)
            
            st.success("Done!")
            for result in results[:3]:  # Top 3
                st.write(f"{result['label']}: {result['score']:.2%}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Run: streamlit run app.py
```

**Pros:** Automatic UI, minimal boilerplate, state management  
**Cons:** Limited control, one user at a time, not suitable for high-load production

---

### Template 2: Flask with API Endpoints

```python
# Save as app.py
from flask import Flask, request, jsonify, render_template
import torch
from transformers import pipeline
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model at startup
model = None

def load_model():
    global model
    model = pipeline("zero-shot-classification", 
                     model="facebook/bart-large-mnli")
    logging.info("Model loaded")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        candidates = data.get('candidates', [])
        
        if not text or not candidates:
            return jsonify({"error": "Missing text or candidates"}), 400
        
        result = model(text, candidates)
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
```

**HTML template (save as templates/index.html):**
```html
<!DOCTYPE html>
<html>
<head>
    <title>ML API</title>
</head>
<body>
    <h1>Deep Learning Prediction</h1>
    <textarea id="text" placeholder="Enter text..."></textarea>
    <input type="text" id="candidates" placeholder="Candidate labels (comma-separated)">
    <button onclick="predict()">Predict</button>
    <div id="result"></div>

    <script>
        function predict() {
            const text = document.getElementById('text').value;
            const candidates = document.getElementById('candidates').value.split(',')
                .map(s => s.trim());
            
            fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text, candidates})
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('result').innerHTML = 
                    '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            })
            .catch(e => console.error(e));
        }
    </script>
</body>
</html>
```

**Pros:** Full control, multiple users, API-based (reusable)  
**Cons:** More boilerplate, requires frontend knowledge

---

### Template 3: FastAPI (Modern Python)

```python
# Save as app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from transformers import pipeline
from PIL import Image
import io

app = FastAPI(title="ML API")

class TextInput(BaseModel):
    text: str
    candidates: list[str]

# Load model at startup
@app.on_event("startup")
async def load_model():
    global classifier
    classifier = pipeline("zero-shot-classification", 
                         model="facebook/bart-large-mnli")

@app.post("/api/predict")
async def predict(input_data: TextInput):
    try:
        result = classifier(input_data.text, input_data.candidates)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Your preprocessing and inference here
        
        return {"filename": file.filename, "prediction": "..."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run: uvicorn app:app --reload
```

**Pros:** Async support, automatic API docs (Swagger UI at `/docs`), fast, modern  
**Cons:** Requires FastAPI + Uvicorn knowledge

---

## Common Pitfalls and Solutions

### Pitfall 1: Model Loads on Every Prediction
**Problem:** App is slow, model reloads every request

**Solution:** Load model at startup (Streamlit: `@st.cache_resource`, Flask: module-level, FastAPI: `@app.on_event("startup")`)

### Pitfall 2: Preprocessing Mismatches
**Problem:** Model trained on normalized [0, 1] but app passes [0, 255]

**Solution:**
- Document exact preprocessing in technical report
- Test preprocessing separately before integration
- Print shapes at every step during development

### Pitfall 3: Unhandled Edge Cases
**Problem:** App crashes on invalid input

**Solution:** Validate before preprocessing
```python
if not 0 <= user_value <= 100:
    return {"error": "Value must be between 0 and 100"}
```

### Pitfall 4: Missing Dependencies
**Problem:** Submission works locally but fails when grader installs

**Solution:** 
```bash
pip freeze > requirements.txt  # Or use pipdeptree
```
Test install in fresh environment:
```bash
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
python app.py
```

### Pitfall 5: Hardcoded Paths
**Problem:** `/Users/alice/models/model.pth` doesn't exist on grader's machine

**Solution:** Use relative paths
```python
# ✗ Bad
model_path = "/Users/alice/assignment2/model.pth"

# ✓ Good
import os
model_path = os.path.join(os.path.dirname(__file__), "model.pth")
```

---

## Deployment Considerations (Bonus)

### Thought Experiment: Production Readiness

Even though this is a local prototype, consider:

1. **Concurrent requests:** Can multiple users access simultaneously?
   - Streamlit: No (single-threaded)
   - Flask/FastAPI: Yes (with proper setup)

2. **Memory usage:** Does model stay in memory or reload?
   - Should always stay loaded (cache at startup)

3. **Timeout:** What if inference is very slow?
   - Add timeout: `result = model(data, timeout=30)`

4. **Input validation:** Do you check input before sending to model?
   - Always validate; display helpful errors

5. **Scalability:** Could you handle 100 requests/second?
   - Probably not with a single machine
   - Consider: load balancing, async workers (Gunicorn + Flask), GPU inference

### Example: Deploying with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
This runs 4 worker processes and handles multiple requests.

---

## Recommended Timeline

### Day 1 (2 hours)
- [ ] Choose model + domain
- [ ] Verify model loads and runs in Jupyter notebook
- [ ] Test preprocessing with sample data

### Day 2 (2–2.5 hours)
- [ ] Build web interface (Streamlit or Flask template)
- [ ] Integrate model into app
- [ ] Test with 5+ examples (good and bad inputs)

### Day 3 (1–1.5 hours)
- [ ] Add error handling
- [ ] Write README with setup instructions
- [ ] Create technical report
- [ ] Test clean install: `pip install -r requirements.txt && python app.py`

### Day 4 (polish, if time)
- [ ] Add sample data and screenshots
- [ ] Optimize UI
- [ ] Consider deployment bonus

---

## Submission Sanity Check

Before submitting, verify:

```bash
# 1. Extract ZIP (not inside)
unzip assignment_2_submission.zip
cd assignment_2_submission

# 2. Fresh Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run app
python app.py

# 5. Test with sample input
# (Use browser or curl, verify output is sensible)

# 6. Check file structure
ls -la  # Should see: app.py, README.md, TECHNICAL_REPORT.md, requirements.txt
```

---

## Resources Summary

| Topic | Resource | Time to Learn |
|-------|----------|----------------|
| Streamlit | https://streamlit.io/docs | 30 min |
| Flask | https://flask.palletsprojects.com/quickstart | 1 hour |
| FastAPI | https://fastapi.tiangolo.com/tutorial | 1.5 hours |
| Hugging Face Hub | https://huggingface.co/docs | 30 min |
| PyTorch basics | https://pytorch.org/tutorials | 2–3 hours |
| Model debugging | Course lecture notes (Week 3) | Review |

---

**Questions? Post on course forum or visit office hours.**

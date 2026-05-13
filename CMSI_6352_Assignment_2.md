# CMSI 6352 — Deep Learning

## Assignment #2: Rapidly Prototyping a Deep Learning Model in Production

**Due:** End of Week 4  
**Total Points:** 100  

---

## Overview

In this assignment, you will build a **locally-hosted web application** that takes user input, performs a deep learning prediction or analysis, and displays results. The goal is to bridge the gap between standalone model development and real-world deployment by teaching you how to integrate neural networks into production-like interfaces.

You will practice:
- Packaging a trained or pre-trained deep learning model
- Building a simple HTTP API or web interface
- Handling user input and model inference
- Debugging inference pipelines
- Communicating results clearly

---

## Learning Objectives

By completing this assignment, you will be able to:

1. **Integrate deep learning models** into web applications
2. **Design user-friendly input/output interfaces** for neural network predictions
3. **Optimize inference speed** and manage model memory constraints
4. **Diagnose and fix common deployment issues** (input preprocessing, batch handling, error handling)
5. **Document and communicate** technical choices clearly

---

## Requirements

### Core Requirements (All must be completed)

#### 1. **Select a Model and Domain**

Choose **one** of the following categories (or propose your own for instructor approval):

- **Image Classification:** CIFAR-10, MNIST, or a fine-tuned ImageNet classifier
- **Sentiment Analysis / NLP:** Text classification (positive/negative/neutral reviews, toxicity detection)
- **Time Series Prediction:** Simple regression (stock prices, temperature forecasting)
- **Tabular Data Prediction:** Regression or classification on structured data (e.g., housing prices, Iris classification)
- **Audio Classification:** Audio event detection or simple speech command recognition
- **Generative:** Image generation (DCGAN, diffusion model inference), text generation (GPT-like, character-level RNN)

**Rationale:** You may use a **pre-trained model** (e.g., from TensorFlow Hub, PyTorch Model Zoo, Hugging Face) OR train your own model. Pre-trained is acceptable and encouraged for this assignment—the focus is integration, not model training.

#### 2. **Build a Locally-Hosted Web Interface**

The application must:

- **Accept user input** via a web form or API endpoint (at minimum, a GET or POST request)
- **Perform model inference** on submitted input
- **Return results** clearly displayed in the browser or as JSON
- **Run locally** without requiring cloud services (localhost:PORT)
- **Be accessible via a browser or command-line client** (e.g., `curl`)

**Technology Stack Options** (choose one or more):

| Framework | Pros | Cons | Difficulty |
|-----------|------|------|-----------|
| **Flask** (Python) | Lightweight, easy to learn, great for prototyping | Minimal built-in features | Easy |
| **FastAPI** (Python) | Fast, automatic API docs (Swagger), modern Python | Steeper learning curve than Flask | Medium |
| **Streamlit** (Python) | Fastest prototyping, automatic UI generation, zero frontend knowledge needed | Less control over UI, slower inference | Easy |
| **Django** (Python) | Full-featured web framework | Overkill for this assignment | Hard |
| **Node.js + Express** (JavaScript) | Good if you prefer JavaScript backend | Requires Node.js; more boilerplate | Medium |

**Recommendation:** Use **Streamlit** or **Flask** for simplicity. If you use Streamlit, you must also include a brief explanation of why API-based approaches (Flask/FastAPI) are preferred in production.

#### 3. **Preprocessing and Inference Pipeline**

Your application must:

- **Accept raw user input** (images, text, tabular data, audio)
- **Preprocess input correctly** (normalize, resize, tokenize, etc.)
- **Run model inference**
- **Post-process outputs** (decode predictions, format probabilities, etc.)
- **Handle errors gracefully** (invalid input, model failures, etc.)

**Example:**
```
User uploads JPEG image
  → Read image file
  → Resize to [224, 224]
  → Normalize to [0,1]
  → Convert to tensor
  → Run model
  → Get predicted class + confidence
  → Display "Predicted: Dog (92% confidence)"
```

#### 4. **Input Validation and Error Handling**

Your application must robustly handle:

- Invalid input types (e.g., uploading text when expecting image)
- Out-of-range or malformed data
- Missing required fields
- Model errors (NaN outputs, out-of-memory, etc.)

Display **user-friendly error messages**, not raw Python tracebacks.

#### 5. **Code Organization and Documentation**

Your submission must include:

- **Main application file** (e.g., `app.py`, `main.py`)
- **Model loading and inference code** (separate module if possible)
- **README.md** with:
  - Instructions to install dependencies and run the app
  - Description of the model and domain
  - Example usage
  - How to provide sample input (screenshots, sample files)
- **requirements.txt** or **environment.yml** with all dependencies
- **Well-commented code** explaining key preprocessing/inference steps

#### 6. **Model Performance Reporting**

Include a **brief technical report** (1-2 pages) documenting:

- **Model Architecture:** What model did you use? Pre-trained or custom?
- **Input/Output Specification:** What does the model expect? What does it return?
- **Preprocessing Details:** Exact normalization, resizing, tokenization steps
- **Inference Speed:** How long does a prediction take? (e.g., "~150ms per image on CPU")
- **Known Limitations:** What cases does the model struggle with? What are edge cases?

---

## Deliverables

Submit a **ZIP file** containing:

```
assignment_2_submission.zip
├── app.py                          # Main application
├── model.py                        # Model loading and inference
├── preprocess.py                   # (Optional) Preprocessing utilities
├── requirements.txt                # Dependencies
├── README.md                       # Setup and usage instructions
├── TECHNICAL_REPORT.md            # 1-2 page technical documentation
├── sample_input/                   # (Optional) Sample data for testing
│   ├── image.jpg
│   ├── text.txt
│   └── ...
└── screenshots/                    # (Optional but encouraged) Screenshots of UI
    ├── input_form.png
    ├── prediction_result.png
    └── error_handling.png
```

---

## Grading Rubric (100 points)

| Category | Points | Criteria |
|----------|--------|----------|
| **Functionality** | 35 | App runs locally without errors; accepts input; performs inference; displays results clearly |
| **Preprocessing** | 15 | Input correctly normalized/resized/formatted; no data leakage; handles edge cases |
| **Error Handling** | 10 | Gracefully handles invalid input; displays helpful error messages; never crashes on bad input |
| **Code Quality** | 15 | Well-organized, commented, modular (not one giant file); follows naming conventions |
| **Documentation** | 15 | README is clear and complete; technical report explains model choice and performance; setup instructions work as written |
| **Bonus** | +5 | Exceptional UX; multiple model options; interesting domain choice; advanced deployment consideration (e.g., batch inference, caching) |

---

## Tips and Best Practices

### Model Selection
- Start with **pre-trained models**—they're faster and often more accurate than quickly trained custom models.
- Hugging Face, TensorFlow Hub, PyTorch Model Zoo, and Scikit-Learn have excellent pre-trained options.
- If training your own model, keep it simple: MNIST, CIFAR-10, or a small custom dataset.

### Development Workflow
1. **Start with a Jupyter notebook** to verify your preprocessing and model loading work correctly
2. **Move to a script** (e.g., `app.py`) once the notebook is solid
3. **Test locally** with sample data before submission
4. **Time your inference** to report realistic latency

### Debugging Tips
- **Print shape information** at every preprocessing step to catch dimension mismatches
- **Save and inspect intermediate outputs** (normalized image, embeddings, logits) during development
- **Use a small test set** during development for fast iteration
- **Separately verify model loading** works before integrating into the web app

### Inference Optimization
- **Batch inference:** If building an API that handles multiple requests, consider batching predictions
- **Model caching:** Load the model once at startup, not on every prediction
- **GPU vs. CPU:** Mention whether your inference runs on GPU (if available) or CPU, and note the difference

### Production Readiness (Bonus Consideration)
- **Concurrent requests:** Can your app handle multiple simultaneous requests? (Streamlit apps handle one user at a time)
- **Timeout handling:** What happens if model inference is slow?
- **Stateless design:** Can the app be restarted without losing state?

---

## Example Project Scope

### Image Sentiment Classification (Good Scope)
- Use Hugging Face Vision Transformer for image-to-text, or ImageNet classifier
- Web form accepts image upload
- Displays predicted class + confidence
- ~2–3 hours of work

### Toxic Comment Detection (Good Scope)
- Use pre-trained BERT from Hugging Face Transformers
- Web form accepts text input
- Returns toxicity probability and explanation
- ~2 hours of work

### Custom CIFAR-10 Classifier (Good Scope if Training)
- Train a simple CNN on CIFAR-10 (from scratch or transfer learning)
- Upload image → prediction
- ~3–4 hours of work

### Full Multimodal System (Probably Too Broad)
- Build a system that handles 5 different input types
- Train multiple models from scratch
- Deploy to cloud
→ **Scope creep.** Stick to one domain.

---

## Questions to Ask Before Starting

1. **What data will users provide?** (images, text, numbers, audio?)
2. **What should the model predict?** (class label, score, generated content?)
3. **How will I verify it's working?** (manual testing with sample data?)
4. **What framework is easiest for me?** (Flask for flexibility, Streamlit for speed?)
5. **How will I handle bad input?** (validation before preprocessing?)

---

## Submission Checklist

- [ ] App runs locally with `python app.py` or similar command
- [ ] README includes installation and run instructions
- [ ] Model loads without errors
- [ ] At least 3 test cases with different inputs (success and failure cases)
- [ ] Code is commented and organized
- [ ] Technical report (1-2 pages) includes model architecture, preprocessing, and performance notes
- [ ] requirements.txt / environment.yml is complete
- [ ] No hardcoded paths (e.g., `/Users/yourname/...`)
- [ ] ZIP file is organized as specified above

---

## Resources

### Web Frameworks
- **Streamlit:** https://streamlit.io/ (quickest to prototype)
- **Flask:** https://flask.palletsprojects.com/ (lightweight, traditional)
- **FastAPI:** https://fastapi.tiangolo.com/ (fast, async-friendly)

### Pre-trained Models
- **Hugging Face Hub:** https://huggingface.co/ (NLP, vision, audio)
- **TensorFlow Hub:** https://tfhub.dev/ (images, text, video, audio)
- **PyTorch Model Zoo:** https://pytorch.org/vision/stable/models.html (computer vision)
- **OpenAI/GitHub:** CLIP, DALL-E, GPT-2 (multimodal, generative)

### Model Libraries
- **Transformers (Hugging Face):** NLP and vision models
- **TensorFlow/Keras:** General deep learning
- **PyTorch:** General deep learning
- **Scikit-Learn:** Simpler ML models, preprocessing utilities

### Examples & Tutorials
- **Streamlit gallery:** https://streamlit.io/gallery
- **Flask + ML tutorial:** Miguel Grinberg's Flask Mega-Tutorial (free online)
- **FastAPI + ML example:** Official FastAPI tutorial with ML examples

---

## Late Submission Policy

- **On time:** Full credit
- **1 day late:** -10%
- **2 days late:** -20%
- **>2 days late:** Contact instructor

---

## Getting Help

- **Office hours:** Ask about debugging deployment issues, framework recommendations
- **Piazza/Course Discussion:** Post about common errors or framework questions
- **Instructor email:** For scope questions or infrastructure issues

---

**Good luck! This assignment bridges an important gap between theory and practice. Your submission should be a working tool, not perfection.**

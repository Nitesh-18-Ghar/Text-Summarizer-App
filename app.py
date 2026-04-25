from fastapi import FastAPI, Request      # Requests se jitne v client side se req. aayega unko handle karna 
from pydantic import BaseModel          # Data validation ke liye(Text mein hi hona chahiye, aisa nhi ki 3, 4 numbers mein likh diya)
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates     # For showing the UI In multiple Pages
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles 

# Initialize Our FastAPI App
app = FastAPI(title="Text summarizer App", description="This Is The Text Summarizxation App Using T5 Model", version="1.0.0")

# Load The T5 Model & Tokenizer
model = T5ForConditionalGeneration.from_pretrained("./Saved_Summary_Models")
tokenizer = T5Tokenizer.from_pretrained("./Saved_Summary_Models")

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# Templates --> Means Hmare UI(HTML File) kahan Exist kiya hai
templates = Jinja2Templates(directory=".")

# Input Schema(Format) for dialogue => String
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) 
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text) 
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue : str) -> str:     # Jo summarize dialogue aayega wo bhi str mein hoga 
    dialogue = clean_data(dialogue)

    # Tokenize
    inputs = tokenizer(
        dialogue,
        padding = "max_length",
        max_length = 512,
        truncation = True,
        return_tensors="pt"
    ).to(device)

    # Generate The Summary => Token IDs
    model.to(device) 
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,
        early_stopping = True
    )

    # Decode The Summary
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

# API Endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.post("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
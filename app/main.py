from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import time

app = FastAPI(
    title       = "LLM Finetuning Pipeline API",
    description = "LLaMA 3.2-1B fine-tuned with QLoRA + DPO",
    version     = "1.0.0"
)

MODEL_ID = "pranav6905/llama-1b-sft-dpo-final"  

# Load once at startup
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype       = torch.float32,
    low_cpu_mem_usage = True,
)
pipe = pipeline(
    "text-generation",
    model     = model,
    tokenizer = tokenizer,
)
print("✅ Model ready")

# ── Schemas ───────────────────────────────────────────────
class GenerateRequest(BaseModel):
    instruction:    str
    max_new_tokens: int   = 200
    temperature:    float = 0.7

class GenerateResponse(BaseModel):
    instruction:     str
    response:        str
    latency_seconds: float
    model_id:        str

# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status"  : "running",
        "model"   : MODEL_ID,
        "endpoints": ["/generate", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if not request.instruction.strip():
        raise HTTPException(status_code=400, detail="Instruction cannot be empty")

    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>user<|end_header_id|>\n{request.instruction}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    try:
        start  = time.time()
        output = pipe(
            prompt,
            max_new_tokens     = request.max_new_tokens,
            temperature        = request.temperature,
            do_sample          = True,
            repetition_penalty = 1.1,
            return_full_text   = False,
        )
        latency  = round(time.time() - start, 2)
        response = output[0]["generated_text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(
        instruction     = request.instruction,
        response        = response,
        latency_seconds = latency,
        model_id        = MODEL_ID,
    )
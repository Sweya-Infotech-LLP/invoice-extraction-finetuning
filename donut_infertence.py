import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Donut Model Inference API",
    description="Dedicated API for Donut model inference with file upload",
    version="1.0.0"
)

# Global variables for model and processor
processor = None
model = None
device = None

def load_model():
    """Load the Donut model and processor"""
    global processor, model, device
    
    try:
        logger.info("Loading Donut model and processor...")
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        logger.info(f"Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    if not load_model():
        raise Exception("Failed to load Donut model")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Donut Model Inference API",
        "endpoints": {
            "inference": "POST /infer",
            "health": "GET /health",
            "model_info": "GET /model-info"
        },
        "model": "naver-clova-ix/donut-base"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "naver-clova-ix/donut-base",
        "device": device,
        "model_loaded": model is not None,
        "processor_loaded": processor is not None
    }

@app.post("/infer")
async def infer_invoice(
    file: UploadFile = File(...),
    task_prompt: str = "<s_invoice>",
    max_length: int = 512
):
    """
    Perform inference on uploaded invoice image using Donut model
    
    Args:
        file: Uploaded image file (PNG, JPG, JPEG)
        task_prompt: Task prompt for the model (default: "<s_invoice>")
        max_length: Maximum length of generated text (default: 512)
        
    Returns:
        JSON response with inference results
    """
    try:
        # Validate file type
        allowed_extensions = {'.png', '.jpg', '.jpeg'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        logger.info(f"Processing inference request for file: {file.filename}")
        
        # Read and process the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Prepare image for model
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        
        # Set up task prompt
        decoder_input_id = processor.tokenizer.convert_tokens_to_ids(task_prompt)
        
        # Perform inference
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_start_token_id=decoder_input_id,
                max_length=max_length,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        # Decode the output
        prediction = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Save response with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_data = {
            "filename": file.filename,
            "task_prompt": task_prompt,
            "max_length": max_length,
            "prediction": prediction,
            "timestamp": timestamp,
            "device_used": device
        }
        
        # Save to file (optional)
        response_file = f"inference_response_{timestamp}.json"
        try:
            import json
            with open(response_file, 'w') as f:
                json.dump(response_data, f, indent=2)
            response_data["response_file"] = response_file
        except Exception as e:
            logger.warning(f"Could not save response file: {e}")
        
        logger.info(f"Inference completed successfully for {file.filename}")
        return JSONResponse(content=response_data, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
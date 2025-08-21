from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import shutil
import tempfile
from typing import Optional
import logging
from pydantic import BaseModel
from datetime import datetime

from training import DonutTrainer
from inference import InvoiceProcessor
from config import Config
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import io

# Configure logging to both file and console
def setup_logger(name: str, log_file: str):
    """Setup logger with both file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Setup app logger
app_logger = setup_logger(
    "fastapi_app", 
    os.path.join(Config.LOGS_DIR, "app.log")
)

app = FastAPI(
    title="Donut Invoice Processing API",
    description="API for training Donut models (with local files or uploads) and processing invoices with RAG and LLM",
    version="1.0.0"
)

# Pydantic models for request/response
class TrainingRequest(BaseModel):
    dataset_path: str
    num_epochs: Optional[int] = 5

class FileTrainingRequest(BaseModel):
    num_epochs: Optional[int] = 5

class TrainingResponse(BaseModel):
    status: str
    message: str

class ProcessingResponse(BaseModel):
    status: str
    donut_output: Optional[str] = None
    retrieved_documents: Optional[list] = None
    structured_response: Optional[str] = None
    donut_response_file: Optional[str] = None
    llm_response_file: Optional[str] = None
    invoice_type: Optional[str] = None
    message: Optional[str] = None

# Global instances
donut_trainer = None
invoice_processor = None

# Donut inference model and processor
donut_processor = None
donut_model = None
donut_device = None

def load_donut_model():
    """Load the Donut model and processor for inference"""
    global donut_processor, donut_model, donut_device
    
    try:
        app_logger.info("Loading Donut model and processor for inference...")
        donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        
        donut_device = "cuda" if torch.cuda.is_available() else "cpu"
        donut_model.to(donut_device)
        
        app_logger.info(f"Donut inference model loaded successfully on {donut_device}")
        return True
        
    except Exception as e:
        app_logger.error(f"Error loading Donut inference model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize global instances on startup"""
    global donut_trainer, invoice_processor
    
    # Validate configuration and create directories
    try:
        Config.validate_config()
        app_logger.info("Configuration validated and directories created successfully")
    except Exception as e:
        app_logger.error(f"Configuration validation failed: {e}")
        raise
    
    donut_trainer = DonutTrainer()
    invoice_processor = InvoiceProcessor()
    
    # Load Donut inference model
    load_donut_model()
    
    app_logger.info("Application started successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Donut Invoice Processing API",
        "endpoints": {
            "train": "POST /train-donut",
            "train_upload": "POST /train-donut-upload",
            "infer_donut": "POST /infer-donut",
            "process": "POST /process-invoice",
            "health": "GET /health",
            "model_status": "GET /model-status",
            "folder_structure": "GET /folder-structure"
        },
        "folder_structure": {
            "models": Config.MODELS_DIR,
            "logs": Config.LOGS_DIR,
            "responses": Config.RESPONSES_DIR,
            "llm_responses": Config.LLM_RESPONSES_DIR
        }
    }

@app.post("/train-donut", response_model=TrainingResponse)
async def train_donut_model(request: TrainingRequest):
    """
    Train and save a Donut model with a local dataset file path
    
    Args:
        request: TrainingRequest containing local dataset path and number of epochs
        
    Returns:
        TrainingResponse with training status and message
    """
    try:
        app_logger.info(f"Starting Donut model training with dataset: {request.dataset_path}")
        
        # Validate dataset path
        if not os.path.exists(request.dataset_path):
            raise HTTPException(
                status_code=400, 
                detail=f"Dataset path does not exist: {request.dataset_path}"
            )
        
        # Train the model
        result = donut_trainer.train_and_save(
            dataset_path=request.dataset_path,
            num_epochs=request.num_epochs
        )
        
        if result["status"] == "success":
            app_logger.info("Training completed successfully")
            return TrainingResponse(**result)
        else:
            app_logger.error(f"Training failed: {result['message']}")
            raise HTTPException(status_code=500, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Unexpected error during training: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during training: {str(e)}"
        )

@app.post("/train-donut-upload", response_model=TrainingResponse)
async def train_donut_model_with_upload(
    file: UploadFile = File(...),
    num_epochs: Optional[int] = Form(5)
):
    """
    Train and save a Donut model with an uploaded dataset file
    
    Args:
        file: Uploaded dataset file (JSON)
        num_epochs: Number of training epochs (default: 5)
        
    Returns:
        TrainingResponse with training status and message
    """
    try:
        app_logger.info(f"Starting Donut model training with uploaded file: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith('.json'):
            raise HTTPException(
                status_code=400,
                detail="Only JSON dataset files are supported"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Train the model with the uploaded file
            result = donut_trainer.train_and_save(
                dataset_path=temp_file_path,
                num_epochs=num_epochs
            )
            
            if result["status"] == "success":
                app_logger.info("Training completed successfully")
                return TrainingResponse(**result)
            else:
                app_logger.error(f"Training failed: {result['message']}")
                raise HTTPException(status_code=500, detail=result["message"])
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Unexpected error during training: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during training: {str(e)}"
        )

@app.post("/process-invoice", response_model=ProcessingResponse)
async def process_invoice(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    invoice_type: Optional[str] = Form("utility")
):
    """
    Process an uploaded invoice through Donut, RAG, and LLM pipeline
    
    Args:
        file: Uploaded invoice file (PDF or image)
        query: Optional query for RAG retrieval
        invoice_type: Type of invoice ("utility" or "telecom") for JSON extraction
        
    Returns:
        ProcessingResponse with complete processing results
    """
    try:
        app_logger.info(f"Processing invoice file: {file.filename} with type: {invoice_type}")
        
        # Validate file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.pdf'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Validate invoice type
        valid_invoice_types = ["utility", "telecom"]
        if invoice_type not in valid_invoice_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid invoice type. Allowed: {', '.join(valid_invoice_types)}"
            )
        
        # Check if trained model exists
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            raise HTTPException(
                status_code=400,
                detail="Trained Donut model not found. Please train the model first using /train-donut endpoint."
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Process the invoice
            result = invoice_processor.process_invoice(
                image_path=temp_file_path,
                query=query,
                invoice_type=invoice_type
            )
            
            if result["status"] == "success":
                app_logger.info("Invoice processing completed successfully")
                return ProcessingResponse(**result)
            else:
                app_logger.error(f"Invoice processing failed: {result['message']}")
                raise HTTPException(status_code=500, detail=result["message"])
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Unexpected error during invoice processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during invoice processing: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/model-status")
async def model_status():
    """Check if trained model exists"""
    model_exists = os.path.exists(Config.MODEL_SAVE_PATH)
    
    return {
        "model_exists": model_exists,
        "model_path": Config.MODEL_SAVE_PATH if model_exists else None,
        "message": "Model is ready for inference" if model_exists else "Model not found - please train first"
    }

@app.post("/infer-donut")
async def infer_donut(
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
        # Check if Donut model is loaded
        if donut_processor is None or donut_model is None:
            raise HTTPException(
                status_code=500,
                detail="Donut model not loaded. Please restart the server."
            )
        
        # Additional model validation
        app_logger.info(f"Model loaded: {donut_model is not None}")
        app_logger.info(f"Processor loaded: {donut_processor is not None}")
        app_logger.info(f"Model device: {donut_device}")
        app_logger.info(f"Model parameters count: {sum(p.numel() for p in donut_model.parameters())}")
        
        # Validate file type
        allowed_extensions = {'.png', '.jpg', '.jpeg'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        app_logger.info(f"Processing Donut inference request for file: {file.filename}")
        
        # Read and process the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Prepare image for model
        pixel_values = donut_processor(image, return_tensors="pt").pixel_values.to(donut_device)
        
        # Set up task prompt - try different approaches
        try:
            decoder_input_id = donut_processor.tokenizer.convert_tokens_to_ids(task_prompt)
            app_logger.info(f"Task prompt '{task_prompt}' converted to ID: {decoder_input_id}")
        except Exception as e:
            app_logger.warning(f"Could not convert task prompt '{task_prompt}' to ID: {e}")
            # Fallback to empty string
            decoder_input_id = donut_processor.tokenizer.convert_tokens_to_ids("")
            app_logger.info(f"Using fallback empty string, ID: {decoder_input_id}")
        
        # Perform inference
        donut_model.eval()
        
        # Debug: Print input shapes and device
        app_logger.info(f"Input pixel_values shape: {pixel_values.shape}")
        app_logger.info(f"Input device: {pixel_values.device}")
        app_logger.info(f"Model device: {next(donut_model.parameters()).device}")
        app_logger.info(f"Decoder input ID: {decoder_input_id}")
        
        with torch.no_grad():
            outputs = donut_model.generate(
                pixel_values,
                decoder_start_token_id=decoder_input_id,
                max_length=max_length,
                early_stopping=False,  # Disable early stopping for greedy
                pad_token_id=donut_processor.tokenizer.pad_token_id,
                num_beams=1,  # Use greedy decoding
                do_sample=False,  # Disable sampling
                repetition_penalty=1.2,  # Prevent repetition
                length_penalty=1.0,  # Neutral length penalty
                no_repeat_ngram_size=3  # Prevent 3-gram repetition
            )
        
        # Debug: Print output shapes and analyze tokens
        app_logger.info(f"Generated outputs shape: {outputs.shape}")
        app_logger.info(f"Generated outputs: {outputs}")
        
        # Analyze the tokens
        unique_tokens = torch.unique(outputs[0]).tolist()
        app_logger.info(f"Unique tokens generated: {unique_tokens}")
        
        # Decode individual tokens for debugging
        for i, token_id in enumerate(outputs[0][:20]):  # First 20 tokens
            token_text = donut_processor.tokenizer.decode([token_id])
            app_logger.info(f"Token {i}: ID={token_id}, Text='{token_text}'")
        
        # Decode the output
        prediction = donut_processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        app_logger.info(f"Raw prediction: '{prediction}'")
        
        # Save response with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_data = {
            "filename": file.filename,
            "task_prompt": task_prompt,
            "max_length": max_length,
            "prediction": prediction,
            "timestamp": timestamp,
            "device_used": donut_device,
            "model": "naver-clova-ix/donut-base"
        }
        
        # Save to file (optional)
        response_file = os.path.join(Config.RESPONSES_DIR, f"donut_inference_{timestamp}.json")
        try:
            with open(response_file, 'w') as f:
                import json
                json.dump(response_data, f, indent=2)
            response_data["response_file"] = response_file
        except Exception as e:
            app_logger.warning(f"Could not save response file: {e}")
        
        app_logger.info(f"Donut inference completed successfully for {file.filename}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error during Donut inference: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Donut inference failed: {str(e)}"
        )

@app.get("/folder-structure")
async def get_folder_structure():
    """Get the current folder structure and status"""
    folders = {
        "models": {
            "path": Config.MODELS_DIR,
            "exists": os.path.exists(Config.MODELS_DIR),
            "donut_model": {
                "path": Config.MODEL_SAVE_PATH,
                "exists": os.path.exists(Config.MODEL_SAVE_PATH)
            },
            "vector_db": {
                "path": Config.VECTOR_DB_PATH,
                "exists": os.path.exists(Config.VECTOR_DB_PATH)
            }
        },
        "logs": {
            "path": Config.LOGS_DIR,
            "exists": os.path.exists(Config.LOGS_DIR)
        },
        "responses": {
            "path": Config.RESPONSES_DIR,
            "exists": os.path.exists(Config.RESPONSES_DIR)
        },
        "llm_responses": {
            "path": Config.LLM_RESPONSES_DIR,
            "exists": os.path.exists(Config.LLM_RESPONSES_DIR)
        }
    }
    
    return {
        "folder_structure": folders,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)

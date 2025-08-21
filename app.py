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
    app_logger.info("Application started successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Donut Invoice Processing API",
        "endpoints": {
            "train": "POST /train-donut",
            "process": "POST /process-invoice"
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

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from typing import List
import json
from datetime import datetime

from inference import LayoutLMv3Inference

# Initialize FastAPI app
app = FastAPI(
    title="LayoutLMv3 Invoice Extraction API",
    description="API for extracting invoice information using LayoutLMv3",
    version="1.0.0"
)

# Initialize LayoutLMv3 model
try:
    layoutlmv3 = LayoutLMv3Inference()
    print("✅ LayoutLMv3 model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading LayoutLMv3 model: {str(e)}")
    layoutlmv3 = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LayoutLMv3 Invoice Extraction API",
        "status": "running",
        "model_loaded": layoutlmv3 is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": layoutlmv3 is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/extract_invoice")
async def extract_invoice(file: UploadFile = File(...)):
    """
    Extract invoice information from uploaded file
    
    Args:
        file: Image or PDF file to process
        
    Returns:
        JSON with extracted invoice information
    """
    if not layoutlmv3:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.bmp'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_extension} not supported. Allowed: {list(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the file
        result = layoutlmv3.predict_invoice_type(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Add file info to result
        result["uploaded_file"] = file.filename
        result["file_size"] = len(content)
        result["file_type"] = file_extension
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/extract_batch")
async def extract_batch(files: List[UploadFile] = File(...)):
    """
    Extract invoice information from multiple files
    
    Args:
        files: List of image or PDF files to process
        
    Returns:
        JSON with extracted information for all files
    """
    if not layoutlmv3:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    temp_files = []
    
    try:
        for file in files:
            # Validate file type
            file_extension = os.path.splitext(file.filename)[1].lower()
            allowed_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.bmp'}
            
            if file_extension not in allowed_extensions:
                results.append({
                    "filename": file.filename,
                    "error": f"Unsupported file type: {file_extension}",
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
                temp_files.append(temp_file_path)
            
            # Process the file
            try:
                result = layoutlmv3.predict_invoice_type(temp_file_path)
                result["uploaded_file"] = file.filename
                result["file_size"] = len(content)
                result["file_type"] = file_extension
                results.append(result)
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return JSONResponse(content={
            "batch_results": results,
            "total_files": len(files),
            "processed_files": len([r for r in results if "error" not in r]),
            "timestamp": datetime.now().isoformat()
        })
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if not layoutlmv3:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": layoutlmv3.model_name,
        "device": layoutlmv3.device,
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        reload=True,
        log_level="info"
    )

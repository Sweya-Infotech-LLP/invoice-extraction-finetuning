import os
import json
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor
)
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayoutLMv3Inference:
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base"):
        """
        Initialize LayoutLMv3 inference
        
        Args:
            model_name: Pre-trained model name from HuggingFace
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.image_processor = None
        
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the LayoutLMv3 model and processor"""
        try:
            logger.info(f"Loading LayoutLMv3 model: {self.model_name}")
            
            # Load processor
            self.processor = LayoutLMv3Processor.from_pretrained(self.model_name)
            
            # Load image processor for image preprocessing
            self.image_processor = LayoutLMv3ImageProcessor.from_pretrained(self.model_name)
            
            # Load model for sequence classification (invoice type detection)
            self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2  # Binary classification: invoice vs non-invoice
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _convert_pdf_to_image(self, pdf_path: str) -> Image.Image:
        """
        Convert PDF to PIL Image
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PIL Image object
        """
        try:
            # Try using pdf2image first (recommended)
            try:
                from pdf2image import convert_from_path
                logger.info("Converting PDF using pdf2image...")
                images = convert_from_path(pdf_path, first_page=1, last_page=1)
                if images:
                    return images[0].convert("RGB")
            except ImportError:
                logger.warning("pdf2image not available, trying alternative methods...")
            
            # Fallback: Try using PyMuPDF (fitz)
            try:
                import fitz  # PyMuPDF
                logger.info("Converting PDF using PyMuPDF...")
                doc = fitz.open(pdf_path)
                page = doc[0]  # Get first page
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                doc.close()
                return img.convert("RGB")
            except ImportError:
                logger.warning("PyMuPDF not available...")
            
            # Final fallback: Try using PIL directly (may not work)
            try:
                logger.info("Trying PIL direct PDF opening...")
                img = Image.open(pdf_path)
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    img.seek(0)  # Go to first frame
                return img.convert("RGB")
            except Exception as e:
                logger.error(f"PIL direct PDF opening failed: {str(e)}")
            
            raise ValueError("Could not convert PDF to image. Please install pdf2image or PyMuPDF.")
            
        except Exception as e:
            logger.error(f"Error converting PDF to image: {str(e)}")
            raise
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess image for LayoutLMv3
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (processed_image, image_info)
        """
        try:
            # Load and convert image
            if image_path.lower().endswith('.pdf'):
                logger.info("Processing PDF file...")
                image = self._convert_pdf_to_image(image_path)
            else:
                logger.info("Processing image file...")
                image = Image.open(image_path).convert("RGB")
            
            # Convert PIL to numpy for OpenCV processing
            image_np = np.array(image)
            
            # Resize image to standard size
            target_size = (224, 224)
            image_resized = cv2.resize(image_np, target_size)
            
            # Convert back to PIL
            image_pil = Image.fromarray(image_resized)
            
            # Process image with LayoutLMv3 processor
            encoding = self.processor(
                image_pil,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move to device
            for key, value in encoding.items():
                if isinstance(value, torch.Tensor):
                    encoding[key] = value.to(self.device)
            
            image_info = {
                "original_size": image.size,
                "processed_size": target_size,
                "image_path": image_path,
                "file_type": "pdf" if image_path.lower().endswith('.pdf') else "image"
            }
            
            return encoding, image_info
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def extract_text_with_ocr(self, image_path: str) -> List[Dict]:
        """
        Extract text and bounding boxes using OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries with text and bounding box information
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use pytesseract for OCR
            try:
                import pytesseract
                # Get text with bounding boxes
                ocr_data = pytesseract.image_to_data(image_rgb, output_type=pytesseract.Output.DICT)
                
                text_boxes = []
                for i in range(len(ocr_data['text'])):
                    if int(ocr_data['conf'][i]) > 30:  # Confidence threshold
                        text = ocr_data['text'][i].strip()
                        if text:  # Only add non-empty text
                            bbox = {
                                'text': text,
                                'bbox': [
                                    ocr_data['left'][i],
                                    ocr_data['top'][i],
                                    ocr_data['left'][i] + ocr_data['width'][i],
                                    ocr_data['top'][i] + ocr_data['height'][i]
                                ],
                                'confidence': ocr_data['conf'][i]
                            }
                            text_boxes.append(bbox)
                
                return text_boxes
                
            except ImportError:
                logger.warning("pytesseract not available, using basic text extraction")
                # Fallback: return basic image info
                return [{
                    'text': 'Image loaded successfully',
                    'bbox': [0, 0, image.shape[1], image.shape[0]],
                    'confidence': 100
                }]
                
        except Exception as e:
            logger.error(f"Error in OCR extraction: {str(e)}")
            return []
    
    def predict_invoice_type(self, image_path: str) -> Dict:
        """
        Predict if the image is an invoice and classify its type
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            encoding, image_info = self.preprocess_image(image_path)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Extract text using OCR
            text_boxes = self.extract_text_with_ocr(image_path)
            
            # Basic invoice field detection
            invoice_fields = self._detect_invoice_fields(text_boxes)
            
            result = {
                "is_invoice": bool(predicted_class),
                "confidence": confidence,
                "invoice_type": "invoice" if predicted_class else "non-invoice",
                "extracted_text": text_boxes,
                "invoice_fields": invoice_fields,
                "image_info": image_info,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _detect_invoice_fields(self, text_boxes: List[Dict]) -> Dict:
        """
        Detect common invoice fields from extracted text
        
        Args:
            text_boxes: List of text boxes with OCR results
            
        Returns:
            Dictionary with detected invoice fields
        """
        invoice_fields = {
            "invoice_number": None,
            "date": None,
            "total_amount": None,
            "vendor_name": None,
            "customer_name": None,
            "items": []
        }
        
        # Keywords for different field types
        invoice_keywords = ["invoice", "bill", "receipt"]
        date_keywords = ["date", "issued", "due"]
        amount_keywords = ["total", "amount", "sum", "due", "$", "€", "£"]
        vendor_keywords = ["from", "vendor", "seller", "company"]
        customer_keywords = ["to", "customer", "buyer", "bill to"]
        
        for box in text_boxes:
            text = box['text'].lower()
            
            # Check for invoice number (usually contains numbers and letters)
            if any(keyword in text for keyword in invoice_keywords):
                if not invoice_fields["invoice_number"]:
                    invoice_fields["invoice_number"] = box['text']
            
            # Check for dates
            elif any(keyword in text for keyword in date_keywords):
                if not invoice_fields["date"]:
                    invoice_fields["date"] = box['text']
            
            # Check for amounts
            elif any(keyword in text for keyword in amount_keywords):
                if not invoice_fields["total_amount"]:
                    invoice_fields["total_amount"] = box['text']
            
            # Check for vendor information
            elif any(keyword in text for keyword in vendor_keywords):
                if not invoice_fields["vendor_name"]:
                    invoice_fields["vendor_name"] = box['text']
            
            # Check for customer information
            elif any(keyword in text for keyword in customer_keywords):
                if not invoice_fields["customer_name"]:
                    invoice_fields["customer_name"] = box['text']
        
        return invoice_fields
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_invoice_type(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                results.append({
                    "error": str(e),
                    "image_path": image_path,
                    "timestamp": datetime.now().isoformat()
                })
        
        return results

def main():
    """Example usage of LayoutLMv3 inference"""
    try:
        # Initialize the model
        layoutlmv3 = LayoutLMv3Inference()
        
        # Example image path (replace with your actual image)
        image_path = "../Solaras-test-18.pdf"  # Adjust path as needed
        
        if os.path.exists(image_path):
            # Make prediction
            result = layoutlmv3.predict_invoice_type(image_path)
            
            # Print results
            print("=== LayoutLMv3 Invoice Extraction Results ===")
            print(json.dumps(result, indent=2, default=str))
            
            # Save results
            output_file = f"layoutlmv3_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"\nResults saved to: {output_file}")
            
        else:
            print(f"Image file not found: {image_path}")
            print("Please provide a valid image path")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

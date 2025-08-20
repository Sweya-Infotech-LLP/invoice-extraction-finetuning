import os
import json
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from typing import Dict, Any, List, Optional
import logging
import chromadb
from chromadb.config import Settings
import openai
from dotenv import load_dotenv
from datetime import datetime
from config import Config

# Import utility JSON functions
from utility_json import extract_invoice_json_utility
from telecom_json import extract_invoice_json_telecom

load_dotenv()

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

# Setup inference logger
inference_logger = setup_logger(
    "donut_inference", 
    os.path.join(Config.LOGS_DIR, "inference.log")
)

class DonutInference:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.MODEL_SAVE_PATH
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_model_loaded = False
        
    def load_trained_model(self) -> bool:
        """Load the trained Donut model and processor"""
        try:
            if not os.path.exists(self.model_path):
                inference_logger.error(f"Model path {self.model_path} does not exist")
                return False
                
            inference_logger.info(f"Loading trained model from {self.model_path}")
            self.processor = DonutProcessor.from_pretrained(self.model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.is_model_loaded = True
            inference_logger.info("Trained model loaded successfully")
            return True
            
        except Exception as e:
            inference_logger.error(f"Error loading trained model: {str(e)}")
            return False
    
    def process_invoice_image(self, image_path: str) -> Optional[str]:
        """Process invoice image through Donut model"""
        try:
            if not self.is_model_loaded:
                if not self.load_trained_model():
                    return None
            
            inference_logger.info(f"Processing invoice image: {image_path}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate prediction
            task_prompt = "<s_invoice>"
            decoder_input_id = self.processor.tokenizer.convert_tokens_to_ids(task_prompt)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    decoder_start_token_id=decoder_input_id,
                    max_length=512,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            prediction = self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            inference_logger.info("Invoice processing completed successfully")
            return prediction
            
        except Exception as e:
            inference_logger.error(f"Error processing invoice image: {str(e)}")
            return None

class RAGPipeline:
    def __init__(self, vector_db_path: str = None):
        self.vector_db_path = vector_db_path or Config.VECTOR_DB_PATH
        self.client = None
        self.collection = None
        self.setup_vector_db()
        
    def setup_vector_db(self):
        """Setup ChromaDB vector database"""
        try:
            os.makedirs(self.vector_db_path, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.vector_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="invoice_documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            inference_logger.info("Vector database setup completed")
            
        except Exception as e:
            inference_logger.error(f"Error setting up vector database: {str(e)}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector database"""
        try:
            if not documents:
                return
                
            texts = [doc.get("text", "") for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            inference_logger.info(f"Added {len(documents)} documents to vector database")
            
        except Exception as e:
            inference_logger.error(f"Error adding documents: {str(e)}")
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            documents = []
            for i in range(len(results["documents"][0])):
                doc = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                }
                documents.append(doc)
            
            inference_logger.info(f"Retrieved {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            inference_logger.error(f"Error retrieving documents: {str(e)}")
            return []

class LLMProcessor:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            inference_logger.warning("OpenAI API key not found. LLM processing will be disabled.")
        
    def generate_response(self, prompt: str, invoice_type: str = "utility") -> Optional[str]:
        """Generate response using OpenAI API with structured JSON output"""
        try:
            if not self.api_key:
                return "LLM processing disabled - API key not configured"
            
            # Determine which JSON extraction function to use based on invoice type
            if invoice_type == "utility":
                # Use utility JSON extraction
                return self._extract_utility_json(prompt)
            elif invoice_type == "telecom":
                # Use telecom JSON extraction
                return self._extract_telecom_json(prompt)
            else:
                # Default to OpenAI for general analysis
                return self._generate_openai_response(prompt)
            
        except Exception as e:
            inference_logger.error(f"Error generating LLM response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _extract_utility_json(self, ocr_text: str) -> str:
        """Extract structured JSON using utility extraction function"""
        try:
            result = extract_invoice_json_utility(ocr_text)
            if "error" in result:
                inference_logger.error(f"Utility JSON extraction error: {result['error']}")
                return f"Error in utility JSON extraction: {result['error']}"
            return json.dumps(result, indent=2)
        except Exception as e:
            inference_logger.error(f"Error in utility JSON extraction: {str(e)}")
            return f"Error in utility JSON extraction: {str(e)}"
    
    def _extract_telecom_json(self, ocr_text: str) -> str:
        """Extract structured JSON using telecom extraction function"""
        try:
            result = extract_invoice_json_telecom(ocr_text)
            if "error" in result:
                inference_logger.error(f"Telecom JSON extraction error: {result['error']}")
                return f"Error in telecom JSON extraction: {result['error']}"
            return json.dumps(result, indent=2)
        except Exception as e:
            inference_logger.error(f"Error in telecom JSON extraction: {str(e)}")
            return f"Error in telecom JSON extraction: {str(e)}"
    
    def _generate_openai_response(self, prompt: str) -> str:
        """Generate response using OpenAI API"""
        try:
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert invoice analyst. Analyze the provided invoice data and retrieved documents to provide comprehensive insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            inference_logger.error(f"Error generating OpenAI response: {str(e)}")
            return f"Error generating OpenAI response: {str(e)}"

class InvoiceProcessor:
    def __init__(self, model_path: str = None):
        self.donut_inference = DonutInference(model_path)
        self.rag_pipeline = RAGPipeline()
        self.llm_processor = LLMProcessor()
        
    def save_response(self, response_data: Dict[str, Any], response_type: str = "donut") -> str:
        """Save response to appropriate folder with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if response_type == "donut":
                folder = Config.RESPONSES_DIR
                filename = f"donut_response_{timestamp}.json"
            elif response_type == "llm":
                folder = Config.LLM_RESPONSES_DIR
                filename = f"llm_response_{timestamp}.json"
            else:
                folder = Config.RESPONSES_DIR
                filename = f"response_{timestamp}.json"
            
            filepath = os.path.join(folder, filename)
            
            with open(filepath, 'w') as f:
                json.dump(response_data, f, indent=2)
            
            inference_logger.info(f"Response saved to: {filepath}")
            return filepath
            
        except Exception as e:
            inference_logger.error(f"Error saving response: {str(e)}")
            return ""
        
    def process_invoice(self, image_path: str, query: str = None, invoice_type: str = "utility") -> Dict[str, Any]:
        """Complete invoice processing pipeline"""
        try:
            # Step 1: Process invoice through Donut
            inference_logger.info("Step 1: Processing invoice through Donut model")
            donut_output = self.donut_inference.process_invoice_image(image_path)
            
            if donut_output is None:
                return {
                    "status": "error",
                    "message": "Failed to process invoice through Donut model"
                }
            
            # Save Donut response
            donut_response_data = {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "donut_output": donut_output,
                "invoice_type": invoice_type
            }
            donut_filepath = self.save_response(donut_response_data, "donut")
            
            # Step 2: RAG retrieval
            inference_logger.info("Step 2: Retrieving relevant documents through RAG")
            if query is None:
                query = "invoice analysis and processing"
            
            retrieved_docs = self.rag_pipeline.retrieve_relevant_documents(query)
            
            # Step 3: LLM processing with structured JSON output
            inference_logger.info("Step 3: Generating structured JSON response")
            
            # Use the appropriate JSON extraction based on invoice type
            structured_response = self.llm_processor.generate_response(donut_output, invoice_type)
            
            # Save LLM response
            llm_response_data = {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "donut_output": donut_output,
                "retrieved_documents": retrieved_docs,
                "structured_response": structured_response,
                "invoice_type": invoice_type
            }
            llm_filepath = self.save_response(llm_response_data, "llm")
            
            # Return complete results
            return {
                "status": "success",
                "donut_output": donut_output,
                "retrieved_documents": retrieved_docs,
                "structured_response": structured_response,
                "donut_response_file": donut_filepath,
                "llm_response_file": llm_filepath,
                "invoice_type": invoice_type
            }
            
        except Exception as e:
            inference_logger.error(f"Error in invoice processing pipeline: {str(e)}")
            return {
                "status": "error",
                "message": f"Processing pipeline failed: {str(e)}"
            }

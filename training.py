import os
import json
import io
import torch
import gc
from PIL import Image
from datasets import Dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Trainer,
    TrainingArguments,
)
from typing import Dict, Any, Optional
import logging
from datetime import datetime
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

# Setup training logger
training_logger = setup_logger(
    "donut_training", 
    os.path.join(Config.LOGS_DIR, "training.log")
)

class DonutTrainer:
    def __init__(self, model_save_path: str = None):
        self.model_save_path = model_save_path or Config.MODEL_SAVE_PATH
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_pretrained_model(self):
        """Load the pretrained Donut model and processor"""
        try:
            training_logger.info("Loading pretrained Donut model...")
            self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
            self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
            
            # Configure model
            task_prompt = "<s_invoice>"
            self.model.config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(task_prompt)
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
            
            self.model.to(self.device)
            training_logger.info("Model loaded successfully")
            return True
        except Exception as e:
            training_logger.error(f"Error loading model: {str(e)}")
            return False
    
    def prepare_dataset(self, dataset_path: str) -> Optional[Dataset]:
        """Prepare the dataset for training"""
        try:
            training_logger.info(f"Loading dataset from {dataset_path}")
            
            if dataset_path.endswith('.json'):
                with open(dataset_path, "r") as f:
                    samples = json.load(f)
            else:
                raise ValueError("Dataset must be a JSON file")
            
            hf_dataset = Dataset.from_list(samples)
            
            def load_image(example):
                """Loads the image from a file path into a PIL Image object."""
                image_path = example["image"]
                example["image"] = Image.open(image_path).convert("RGB")
                return example
            
            hf_dataset = hf_dataset.map(load_image)
            training_logger.info(f"Dataset prepared with {len(hf_dataset)} samples")
            return hf_dataset
            
        except Exception as e:
            training_logger.error(f"Error preparing dataset: {str(e)}")
            return None
    
    def preprocess_data(self, dataset: Dataset) -> Optional[Dataset]:
        """Preprocess the dataset for training"""
        try:
            training_logger.info("Preprocessing dataset...")
            
            def preprocess(example):
                """
                Prepares the data for the model.
                - Resizes image and creates pixel_values.
                - Tokenizes ground_truth to create labels.
                """
                image_bytes = example["image"]["bytes"]
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image = image.resize((384, 384))
                pixel_values = self.processor(image, return_tensors="pt").pixel_values[0]

                labels = self.processor.tokenizer(
                    example["ground_truth"],
                    max_length=512, 
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids[0]

                labels[labels == self.processor.tokenizer.pad_token_id] = -100

                return {
                    "pixel_values": pixel_values,
                    "labels": labels
                }
            
            tokenized_dataset = dataset.map(preprocess, remove_columns=["image", "ground_truth"])
            tokenized_dataset.set_format("torch")
            training_logger.info("Dataset preprocessing completed")
            return tokenized_dataset
            
        except Exception as e:
            training_logger.error(f"Error preprocessing data: {str(e)}")
            return None
    
    def train_model(self, dataset: Dataset, num_epochs: int = 5) -> bool:
        """Train the Donut model"""
        try:
            training_logger.info("Starting model training...")
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            training_args = TrainingArguments(
                output_dir=self.model_save_path,
                per_device_train_batch_size=1,
                num_train_epochs=num_epochs,
                fp16=True,                         
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,       
                optim="adamw_8bit",               
                logging_steps=100,
                report_to="none",
                save_strategy="no",
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
            )
            
            trainer.train()
            training_logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            training_logger.error(f"Error during training: {str(e)}")
            return False
    
    def save_model(self) -> bool:
        """Save the trained model and processor"""
        try:
            training_logger.info(f"Saving model to {self.model_save_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(self.model_save_path, exist_ok=True)
            
            self.model.save_pretrained(self.model_save_path)
            self.processor.save_pretrained(self.model_save_path)
            
            training_logger.info("Model saved successfully")
            return True
            
        except Exception as e:
            training_logger.error(f"Error saving model: {str(e)}")
            return False
    
    def train_and_save(self, dataset_path: str, num_epochs: int = 5) -> Dict[str, Any]:
        """Complete training pipeline"""
        try:
            # Load model
            if not self.load_pretrained_model():
                return {"status": "error", "message": "Failed to load pretrained model"}
            
            # Prepare dataset
            dataset = self.prepare_dataset(dataset_path)
            if dataset is None:
                return {"status": "error", "message": "Failed to prepare dataset"}
            
            # Preprocess data
            processed_dataset = self.preprocess_data(dataset)
            if processed_dataset is None:
                return {"status": "error", "message": "Failed to preprocess dataset"}
            
            # Train model
            if not self.train_model(processed_dataset, num_epochs):
                return {"status": "error", "message": "Training failed"}
            
            # Save model
            if not self.save_model():
                return {"status": "error", "message": "Failed to save model"}
            
            return {
                "status": "success", 
                "message": f"Donut model trained and saved at {self.model_save_path}"
            }
            
        except Exception as e:
            training_logger.error(f"Training pipeline error: {str(e)}")
            return {"status": "error", "message": f"Training pipeline failed: {str(e)}"}

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Folder Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    RESPONSES_DIR = os.path.join(BASE_DIR, "responses")
    LLM_RESPONSES_DIR = os.path.join(BASE_DIR, "llm_responses")
    
    # Model Configuration
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "saved_donut_model")
    VECTOR_DB_PATH = os.path.join(MODELS_DIR, "vector_db")
    
    # API Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Training Configuration
    DEFAULT_EPOCHS = int(os.getenv("DEFAULT_EPOCHS", 5))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 5e-5))
    
    # File upload configuration
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.pdf'}
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.RESPONSES_DIR,
            cls.LLM_RESPONSES_DIR,
            cls.MODEL_SAVE_PATH,
            cls.VECTOR_DB_PATH
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not set. LLM processing will be disabled.")
        
        # Create all necessary directories
        cls.create_directories()
        
        return True

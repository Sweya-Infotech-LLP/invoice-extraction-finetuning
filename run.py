#!/usr/bin/env python3
"""
Startup script for the Donut Invoice Processing API
"""

import uvicorn
import os
import sys
from config import Config

def main():
    """Start the FastAPI application"""
    print("🚀 Starting Donut Invoice Processing API...")
    
    # Validate configuration and create directories
    try:
        Config.validate_config()
        print("✅ Configuration validated and directories created successfully")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)
    
    print(f"📁 Created/verified directories:")
    print(f"   Models: {Config.MODELS_DIR}")
    print(f"   Logs: {Config.LOGS_DIR}")
    print(f"   Responses: {Config.RESPONSES_DIR}")
    print(f"   LLM Responses: {Config.LLM_RESPONSES_DIR}")
    
    print(f"\n🌐 Starting server on {Config.HOST}:{Config.PORT}")
    print("📖 API documentation available at:")
    print(f"   http://{Config.HOST}:{Config.PORT}/docs")
    print(f"   http://{Config.HOST}:{Config.PORT}/redoc")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "app:app",
            host=Config.HOST,
            port=Config.PORT,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

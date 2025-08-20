# Donut Invoice Processing API

A FastAPI-based application that provides two main endpoints:
1. **Training endpoint** for fine-tuning Donut models on invoice datasets
2. **Inference endpoint** for processing invoices through Donut, RAG, and LLM pipeline with structured JSON extraction

## Features

- **Donut Model Training**: Fine-tune pre-trained Donut models on custom invoice datasets
- **Invoice Processing**: Extract structured information from invoice images using trained models
- **RAG Pipeline**: Retrieve relevant documents using vector similarity search
- **Structured JSON Extraction**: Generate structured JSON responses using utility and telecom extraction functions
- **Comprehensive Logging**: File-based logging for all operations
- **Response Storage**: Automatic storage of Donut and LLM responses with timestamps
- **RESTful API**: Clean FastAPI endpoints with proper error handling

## Project Structure

```
fine-tunning_invoice/
├── app.py              # Main FastAPI application
├── training.py         # Donut training logic
├── inference.py        # Inference, RAG, and LLM pipeline
├── config.py           # Configuration management
├── utility_json.py     # Utility invoice JSON extraction
├── telecom_json.py     # Telecom invoice JSON extraction
├── requirements.txt    # Python dependencies
├── run.py             # Startup script
├── test_api.py        # API testing script
├── README.md          # This file
├── models/            # Model storage directory
│   ├── saved_donut_model/  # Trained Donut models
│   └── vector_db/          # ChromaDB vector database
├── logs/              # Log files directory
│   ├── training.log         # Training logs
│   ├── inference.log        # Inference logs
│   └── app.log             # API application logs
├── responses/         # Donut response storage
│   └── donut_response_*.json
└── llm_responses/     # LLM response storage
    └── llm_response_*.json
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fine-tunning_invoice
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   HOST=0.0.0.0
   PORT=8000
   DEFAULT_EPOCHS=5
   ```

## Usage

### Starting the API Server

```bash
python run.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Train Donut Model

**Endpoint**: `POST /train-donut`

**Request Body**:
```json
{
  "dataset_path": "/path/to/your/dataset.json",
  "num_epochs": 5
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Donut model trained and saved at ./models/saved_donut_model"
}
```

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/train-donut" \
     -H "Content-Type: application/json" \
     -d '{"dataset_path": "./donut_dataset.json", "num_epochs": 5}'
```

#### 2. Process Invoice

**Endpoint**: `POST /process-invoice`

**Request**: Multipart form data with:
- `file`: Invoice image file (PNG, JPG, JPEG, PDF)
- `query` (optional): Query for RAG retrieval
- `invoice_type`: Type of invoice ("utility" or "telecom")

**Response**:
```json
{
  "status": "success",
  "donut_output": "Extracted invoice information...",
  "retrieved_documents": [...],
  "structured_response": "Structured JSON output...",
  "donut_response_file": "./responses/donut_response_20241201_143022.json",
  "llm_response_file": "./llm_responses/llm_response_20241201_143022.json",
  "invoice_type": "utility"
}
```

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/process-invoice" \
     -F "file=@invoice.png" \
     -F "query=invoice analysis" \
     -F "invoice_type=utility"
```

#### 3. Additional Endpoints

- `GET /`: API information and available endpoints
- `GET /health`: Health check
- `GET /model-status`: Check if trained model exists
- `GET /folder-structure`: View folder structure and status

### Invoice Types

The API supports two types of invoice processing:

1. **Utility Invoices** (`invoice_type=utility`):
   - Uses `utility_json.py` for structured extraction
   - Extracts detailed utility-specific fields
   - Includes meter information, service details, and billing breakdown

2. **Telecom Invoices** (`invoice_type=telecom`):
   - Uses `telecom_json.py` for structured extraction
   - Extracts telecom-specific fields
   - Includes service plans, usage details, and billing information

### Dataset Format

Your training dataset should be a JSON file with the following structure:

```json
[
  {
    "image": "/path/to/image1.png",
    "ground_truth": "Invoice data in text format..."
  },
  {
    "image": "/path/to/image2.png",
    "ground_truth": "Another invoice data..."
  }
]
```

## Configuration

### Folder Structure

The application automatically creates and manages the following directory structure:

- **`models/`**: Contains trained Donut models and vector database
- **`logs/`**: Contains all log files for debugging and monitoring
- **`responses/`**: Stores Donut model outputs with timestamps
- **`llm_responses/`**: Stores structured LLM responses with timestamps

### Model Settings

- **Model Save Path**: Automatically managed in `./models/saved_donut_model/`
- **Vector DB Path**: Automatically managed in `./models/vector_db/`
- **Training Parameters**: Configurable via environment variables

### LLM Configuration

- **OpenAI API Key**: Required for general LLM processing
- **Groq API Key**: Required for utility/telecom JSON extraction
- **Model**: Uses Groq's llama3-8b-8192 for structured extraction
- **Temperature**: Set to 0.0 for consistent structured outputs

## Response Storage

### Donut Responses

Stored in `./responses/` with filename format: `donut_response_YYYYMMDD_HHMMSS.json`

Contains:
- Timestamp
- Image path
- Donut model output
- Invoice type

### LLM Responses

Stored in `./llm_responses/` with filename format: `llm_response_YYYYMMDD_HHMMSS.json`

Contains:
- Timestamp
- Image path
- Donut output
- Retrieved documents
- Structured JSON response
- Invoice type

## Logging

The application provides comprehensive logging to both console and files:

- **Training Logs**: `./logs/training.log`
- **Inference Logs**: `./logs/inference.log`
- **Application Logs**: `./logs/app.log`

Each log entry includes timestamp, logger name, log level, and detailed message.

## Error Handling

The API includes comprehensive error handling for:
- Missing trained models
- Invalid file types
- Invalid invoice types
- Training failures
- Processing errors
- Configuration issues

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Structure

- **`training.py`**: Contains `DonutTrainer` class for model training
- **`inference.py`**: Contains `InvoiceProcessor`, `RAGPipeline`, and `LLMProcessor` classes
- **`utility_json.py`**: Utility invoice JSON extraction using Groq
- **`telecom_json.py`**: Telecom invoice JSON extraction using Groq
- **`app.py`**: FastAPI application with endpoint definitions
- **`config.py`**: Configuration management and directory creation

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU training
2. **Model Not Found**: Ensure you've trained the model first using `/train-donut`
3. **API Key Errors**: Check your OpenAI and Groq API keys
4. **File Upload Issues**: Verify file format and size limits
5. **Directory Creation**: Ensure the application has write permissions

### Logs

The application provides detailed logging for debugging:
- Training progress and errors
- Inference pipeline steps
- RAG retrieval results
- JSON extraction results
- File operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Check the folder structure endpoint
4. Open an issue on GitHub with detailed information

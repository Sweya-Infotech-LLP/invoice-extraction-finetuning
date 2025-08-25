# LayoutLMv3 Invoice Extraction

This folder contains the implementation of LayoutLMv3 for invoice extraction and document understanding.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd layoutlmv3
pip install -r requirements.txt
```

**Note**: You may also need to install Tesseract OCR for text extraction:
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

### 2. Test the Model

```bash
python test_inference.py
```

### 3. Run the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8001`

## 📁 File Structure

```
layoutlmv3/
├── inference.py          # Main LayoutLMv3 inference class
├── test_inference.py     # Simple test script
├── api.py               # FastAPI web service
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Usage

### Basic Inference

```python
from inference import LayoutLMv3Inference

# Initialize the model
layoutlmv3 = LayoutLMv3Inference()

# Process an image/PDF
result = layoutlmv3.predict_invoice_type("path/to/invoice.pdf")

# View results
print(f"Is Invoice: {result['is_invoice']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Extracted Fields: {result['invoice_fields']}")
```

### API Usage

#### Single File Processing
```bash
curl -X POST "http://localhost:8001/extract_invoice" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@invoice.pdf"
```

#### Batch Processing
```bash
curl -X POST "http://localhost:8001/extract_batch" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@invoice1.pdf" \
     -F "files=@invoice2.pdf"
```

#### Health Check
```bash
curl "http://localhost:8001/health"
```

## 🎯 Features

- **Document Classification**: Identifies if a document is an invoice
- **Text Extraction**: OCR-based text extraction with bounding boxes
- **Field Detection**: Automatically detects common invoice fields:
  - Invoice number
  - Date
  - Total amount
  - Vendor name
  - Customer name
- **Batch Processing**: Process multiple documents at once
- **Web API**: RESTful API for easy integration
- **PDF Support**: Handles both images and PDF files

## 🔍 How It Works

1. **Image Preprocessing**: Resizes and normalizes input images
2. **LayoutLMv3 Processing**: Uses the pre-trained LayoutLMv3 model for document understanding
3. **OCR Extraction**: Extracts text and bounding boxes using Tesseract
4. **Field Detection**: Applies keyword-based rules to identify invoice fields
5. **Result Generation**: Returns structured JSON with extracted information

## ⚠️ Important Notes

- **First Run**: The first time you run this, it will download the LayoutLMv3 model (~1.5GB)
- **GPU Support**: Automatically uses CUDA if available, falls back to CPU
- **Memory**: Requires sufficient RAM for model loading (~4GB recommended)
- **File Types**: Supports PNG, JPG, JPEG, PDF, TIFF, BMP

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Ensure sufficient disk space
   - Try running again

2. **OCR Not Working**
   - Install Tesseract OCR
   - Ensure Tesseract is in your PATH
   - Check file permissions

3. **Memory Issues**
   - Close other applications
   - Use smaller batch sizes
   - Consider using CPU instead of GPU

4. **PDF Processing Errors**
   - Ensure PDF is not corrupted
   - Convert PDF to image if needed
   - Check file size limits

### Error Messages

- **"Model not loaded"**: Check if the model downloaded successfully
- **"File type not supported"**: Use supported file formats
- **"Processing error"**: Check file integrity and model status

## 📊 Performance

- **Model Loading**: ~30-60 seconds (first time)
- **Single Document**: ~2-5 seconds
- **Batch Processing**: ~1-2 seconds per document
- **Memory Usage**: ~2-4GB RAM

## 🔮 Future Enhancements

- Fine-tuning on custom invoice datasets
- Support for more document types
- Improved field extraction accuracy
- Custom field definitions
- Export to various formats (CSV, Excel, etc.)

## 📚 References

- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [HuggingFace LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)

## 🤝 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the error logs
3. Ensure all dependencies are properly installed
4. Test with a simple image first

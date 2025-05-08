# Semiconductor Material Analysis System

An advanced AI-powered system for analyzing semiconductor material properties using multiple LLMs (Gemini and Groq) with RAG capabilities.

## Features

- **Multi-model Analysis**: Uses both Gemini and Groq LLMs for comprehensive material evaluation
- **Multi-threading & Batch Processing**: Parallel processing for efficient analysis
- **Semantic Search**: TF-IDF based context retrieval for relevant document searching
- **Response Caching**: Intelligent caching to reduce API calls and speed up processing
- **Visualization**: Multiple chart types for material property comparisons
- **Material Scoring**: Comprehensive evaluation metrics for semiconductor properties
- **Intelligent Context Management**: Automatic context truncation for token limit handling
- **API Management System**: Rate limiting and quota tracking for multiple API providers

## Requirements

- Python 3.8+
- Required libraries:
  - matplotlib
  - scipy
  - scikit-learn
  - numpy
  - requests
  - PyPDF2

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/semiconductor-ai-cot.git
   cd semiconductor-ai-cot
   ```

2. Install dependencies:
   ```bash
   pip install matplotlib scipy scikit-learn numpy requests PyPDF2
   ```

3. Set up API keys:
   - Configure API keys for Gemini and Groq in the model_manager.py file

## Usage

### Basic Usage

```bash
python evaluate_models.py
```

This will:
1. Process materials from the default CSV file
2. Query material properties against Gemini and Groq models
3. Analyze and score each material
4. Generate visualizations in the results/visualizations directory
5. Save detailed analysis to JSON files

### Test Dependencies

To check if all dependencies are properly installed:

```bash
python test_dependencies.py
```

### Directory Structure

- `data/raw/documents/` - Place semiconductor material PDFs and CSV data here
- `results/` - Output files including analysis results and visualizations
- `cache/` - Response cache storage for efficiency

## Advanced Configuration

The script supports multiple configuration options:

```python
# Number of materials to process (useful for testing)
MAX_RECORDS_TO_PROCESS = 3

# Delay between API calls to avoid rate limits
SLEEP_BETWEEN_RECORDS = 2  

# Number of parallel processing threads
MAX_WORKER_THREADS = 4

# Number of materials to process in each batch
BATCH_SIZE = 2
```

## Key Components

### Material Scoring System

Materials are scored based on:
- Bandgap energy
- Conductivity
- Thermal stability
- Crystal structure
- Target application potential

### Visualization Types

1. **Score Bar Chart**: Overall material evaluation scores
2. **Property Comparison**: Scatter plot comparing bandgap vs conductivity
3. **Radar Chart**: Multi-dimensional property comparison

### Semantic Search

The system uses TF-IDF vectorization and cosine similarity to find relevant information in semiconductor research documents for enhanced context.

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

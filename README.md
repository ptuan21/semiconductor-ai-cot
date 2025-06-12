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
- **Auto Engine Detection**: Optional automatic selection of available API engines

## Requirements

- Python 3.8+
- Required libraries:
  - matplotlib
  - scipy
  - scikit-learn
  - numpy
  - requests
  - PyPDF2
  - python-dotenv

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/semiconductor-ai-cot.git
   cd semiconductor-ai-cot
   ```

2. Install dependencies:
   ```bash
   pip install matplotlib scipy scikit-learn numpy requests PyPDF2 python-dotenv
   ```

3. Set up environment variables:
   - Copy the `.env.example` file to `.env`
   - Configure your API keys and settings in the `.env` file:
     ```
     # API Keys
     GEMINI_API_KEYS=your_key1,your_key2,your_key3
     GROQ_API_KEYS=your_groq_key

     # Engine Configuration
     ENGINES_TO_USE=gemini,groq
     USE_AUTO_ENGINE_DETECTION=False

     # Processing Settings
     MAX_RECORDS_TO_PROCESS=20
     MAX_WORKER_THREADS=2
     BATCH_SIZE=2
     SLEEP_BETWEEN_RECORDS=1
     ```

## Usage

### Basic Usage

```bash
python app.py
```

This will:
1. Process materials from the default CSV file
2. Query material properties against configured LLM engines
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
- `analyzed_images/` - Processed and marked images from analysis

## Advanced Configuration

The system supports multiple configuration options through the `.env` file:

### API Configuration
- Multiple API keys support for load balancing
- Configurable engine selection
- Automatic engine detection option

### Processing Settings
- `MAX_RECORDS_TO_PROCESS`: Number of materials to process
- `SLEEP_BETWEEN_RECORDS`: Delay between API calls
- `MAX_WORKER_THREADS`: Number of parallel processing threads
- `BATCH_SIZE`: Number of materials to process in each batch

### Cache Configuration
- `CACHE_DIRECTORY`: Location for storing API response cache
- `API_SCAN_INTERVAL_MINUTES`: Interval for scanning API availability
- `ENABLE_PERIODIC_API_SCAN`: Toggle automatic API availability checking

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
4. **Heatmap**: Detailed property correlation analysis

### Semantic Search

The system uses TF-IDF vectorization and cosine similarity to find relevant information in semiconductor research documents for enhanced context.

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

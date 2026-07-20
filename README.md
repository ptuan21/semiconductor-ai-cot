# Semiconductor AI CoT Project

This project focuses on leveraging **Large Language Models (LLMs)**, such as **LLaMA 3 - 70B** and **Gemini 1.5 Flash**, to analyze, process, and evaluate data related to semiconductor research. The project includes tools for data generation, question extraction, model evaluation, and result analysis.

---

## Project Structure
. ├── analyze_problems.py # Analyze and clean question-answer datasets ├── analyze_results.py # Analyze evaluation results ├── evaluate_models.py # Evaluate LLMs on specific tasks ├── extract_questions.py # Extract questions and answers from PDF documents ├── llm_models.py # LLM wrapper for text generation ├── model_evaluator.py # Evaluate model performance ├── model_manager.py # Manage LLaMA and Gemini models ├── prompts.py # Predefined prompts for LLMs ├── requirements.txt # Python dependencies ├── trained_model.pkl # Trained machine learning model ├── data/ │ ├── processed/ # Processed datasets │ └── raw/ # Raw datasets and scripts │ ├── data_gen.py # Generate fake materials data │ └── documents/ # Raw documents (e.g., PDFs, CSVs) ├── db/ │ └── questions/ # Question-answer datasets ├── model_cache/ # Cached models for LLaMA and Gemini ├── offload/ # Offloaded model data for memory optimization ├── results/ # Evaluation and analysis results │ ├── analysis_results.csv # Analysis results in CSV format │ └── evaluated_results.json # Evaluation results in JSON format └── README.md # Project documentation

---

## Features

### 1. **Data Generation**
- Generate synthetic datasets for semiconductor materials using `data_gen.py`.
- Example dataset includes properties like `resistivity`, `bandgap`, `structure`, and `target`.

### 2. **Question Extraction**
- Extract questions and answers from PDF documents using `extract_questions.py`.
- Supports automated processing of research papers.

### 3. **Model Management**
- Manage and load **LLaMA 3 - 70B** and **Gemini 1.5 Flash** models using `model_manager.py`.
- Includes GPU memory optimization and retry mechanisms.

### 4. **Model Evaluation**
- Evaluate LLMs on tasks like question answering and text generation using `evaluate_models.py`.
- Analyze results with `analyze_results.py`.

### 5. **Prompt Engineering**
- Use predefined prompts for tasks like summarization, question answering, and idea generation in `prompts.py`.

---

## Datasets

### 1. **Fake Materials Data**
- Located in `data/raw/documents/fake_materials_data_with_labels.csv`.
- Contains:
  - `material_name`: Name of the material (e.g., `CuO`, `Si`, `Ge`).
  - `resistivity`: Electrical resistivity (Ohm-m).
  - `bandgap`: Bandgap energy (eV).
  - `structure`: Crystal structure (e.g., `Cubic`, `Hexagonal`).
  - `target`: Binary label for classification tasks.

### 2. **Question-Answer Data**
- Located in `db/questions/`.
- Files:
  - `cot_questions.json`: Raw question-answer pairs.
  - `cot_questions_clean.json`: Cleaned and processed question-answer pairs.

---

## Setup

### 1. **Clone the Repository**
```bash
git clone https://github.com/<your-username>/semiconductor-ai-cot.git
cd semiconductor-ai-cot
```

### 2. **Install Dependencies**
Install Python dependencies using pip:
```bash
pip install -r requirements.txt
```

### 3. **Set Up Environment Variables**
Create a .env file in the root directory with the following content:
```markdown
GEMINI_API_KEY=<your-gemini-api-key>
LLAMA_MODEL_PATH=<path-to-llama-model>
LLAMA_TOKENIZER_PATH=<path-to-llama-tokenizer>
TRANSFORMERS_CACHE=./model_cache
```

### How to Use:
1. Save this content into your [README.md](http://_vscodecontentref_/3) file located at [README.md](http://_vscodecontentref_/4).
2. Customize placeholders like `<your-username>` and `<your-gemini-api-key>` with your actual information.
3. Commit the changes to your Git repository:
   ```bash
   git add README.md
   git commit -m "Add README.md"
   git push origin main
```

Usage
1. Generate Fake Data
Run the following command to generate synthetic materials data:

python data/raw/data_gen.py
2. Extract Questions from PDFs
Extract questions and answers from PDF documents:

python extract_questions.py --input_dir data/raw/documents --output_file db/questions/cot_questions.json
3. Evaluate Models
Evaluate LLaMA and Gemini models on specific tasks:

python evaluate_models.py
4. Analyze Results
Analyze evaluation results:

python analyze_results.py
Dependencies
Key dependencies are listed in requirements.txt. Install them using:

pip install -r requirements.txt
License
This project is licensed under the MIT License. See the LICENSE file for details.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m "Add new feature").
Push to the branch (git push origin feature-branch).
Open a pull request.
📧 Contact
For questions or support, please contact:

Name: Your Name
Email: your.email@example.com
GitHub: your-username


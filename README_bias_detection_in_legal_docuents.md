
# Bias Detection in Legal Documents


This project uses natural language processing (NLP) techniques and machine learning models to detect bias in legal documents. The implementation leverages preprocessing steps, tokenization, contextual embeddings, and a fine-tuned transformer model for bias detection.

## Prerequisites

Before running the notebook, ensure that the following libraries and dependencies are installed:

1. **Python Libraries**
   - pandas
   - numpy
   - torch
   - nltk
   - spacy
   - transformers
   - matplotlib
   - seaborn
   - tqdm
   - sklearn

2. **Additional Setup**
   - Install and download the required Spacy language model:
     ```bash
     pip install spacy
     python -m spacy download en_core_web_sm
     ```
   - Install the Hugging Face Transformers library:
     ```bash
     pip install transformers
     ```

## How to Run the Notebook

1. Clone or download the repository containing the project notebook (`Nlp_Sem_Project.ipynb`).
2. Ensure you have a Python environment with all prerequisites installed.
3. Open the notebook in your preferred editor (e.g., Jupyter Notebook, Google Colab).
4. Follow the steps in the notebook, which include:
   - Preprocessing the dataset
   - Training the model
   - Evaluating performance metrics
   - Analyzing results

5. Run each cell sequentially to execute the project.

## Project Overview

- **Objective**: Detect and analyze bias in legal documents using state-of-the-art NLP techniques.
- **Methodology**: Combines data preprocessing, contextual embedding (DistilBERT), and fine-tuning for a classification task.
- **Key Features**:
  - Tokenization using Hugging Face Transformers.
  - Contextual embeddings with pre-trained transformer models.
  - Evaluation metrics: Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

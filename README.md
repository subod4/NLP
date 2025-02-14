# NLP project: Abstractive and Extractive Text Summarizer

This repository contains tools and resources for both **abstractive** and **extractive** text summarization. Below is an explanation of each file in this repository and its purpose.

---

## Table of Contents
1. [.gitattributes](#.gitattributes)
2. [Abstractive_summarizer.py](#abstractive_summarizerpy)
3. [README.md](#readmemd)
4. [Text_Summarizer_Extractive.ipynb](#text_summarizer_extractiveipynb)
5. [Text_Summarizer_Abstractive.ipynb](#text_summarizer_abstractiveipynb)
6. [requirements.txt](#requirementstxt)
7. [saved_model.zip](#saved_modelzip)

---

## File Descriptions

### .gitattributes
- Configuration file for handling file attributes in Git.

### Abstractive_summarizer.py
- **Purpose**: This is the main Python script to run the fine-tuned T5 model for abstractive summarization.
- **Details**:
  - Loads the pre-trained T5 model from `saved_model.zip`.
  - Provides a function (`summarize_text`) to generate summaries from input text.
  - Includes an optional Gradio-based web interface for interactive summarization.
- **Usage**:
  ```bash
  python Abstractive_summarizer.py
  ```

### README.md
- Documentation file explaining the project and its components.

### Text_Summarizer_Extractive.ipynb
- **Purpose**: Implements extractive text summarization by selecting key sentences from the input.
- **Details**:
  - Demonstrates text preprocessing, feature extraction, and ranking algorithms.
  - Extracts important sentences from the given text to create a summary.
  - Developed using Google Colab, which may require Colab-specific libraries or configurations.
- **Usage**:
  - Open the notebook in a Jupyter environment or Google Colab.
  - Follow the instructions within the notebook to run the extractive summarizer.

### Text_Summarizer_Abstractive.ipynb
- **Purpose**: Implements abstractive text summarization using the T5 model.
- **Details**:
  - Fine-tunes the T5 model on a dataset (e.g., CNN/DailyMail) for summarization.
  - Includes code for training, evaluation, and inference.
  - Generates a trained model, which is saved as `saved_model.zip`.
- **Usage**:
  - Open the notebook in a Jupyter environment or Google Colab.
  - Follow the instructions within the notebook to train the model and generate `saved_model.zip`.

### requirements.txt
- Contains the list of required Python packages to run the summarization models.
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

### saved_model.zip
- Contains the fine-tuned T5 model for abstractive summarization.
- Used by `Abstractive_summarizer.py` for generating summaries.

---

## Requirements
- Python 3.x
- Jupyter Notebook or Google Colab
- Required libraries mentioned in the notebooks and requirments.txt (install using `pip`).

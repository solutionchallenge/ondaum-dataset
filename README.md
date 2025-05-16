# Ondaum Dataset

This repository manages the dataset used for model fine-tuning and text embedding of the Ondaum AI psychological counseling chatbot.

## Dataset Source

The original dataset is sourced from [AI Hub's Emotional Dialogue Dataset](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=270).

## Repository Structure

- `dataset/`: Original and processed dataset files
  - `finetune_dataset.jsonl`: Processed dataset for model fine-tuning
  - `reclassified_dataset.jsonl`: Processed dataset for text-embedding
- `emotion_dataset_to_text_embedding.py`: Script to convert the original dataset into text embedding format
- `text_embedding_to_fine_tuning.py`: Script to convert text embedding data into fine-tuning format
- `create_tuned_model.py`: Script to create and run model fine-tuning
- `manage_tuned_models.py`: Script to manage (list, check status, delete) fine-tuned models

## Copyright

For copyright interpretation of all datasets included in this repository, please refer to the [ondaum-reference](https://github.com/solutionchallenge/ondaum-reference) repository.

## Prerequisites

- Python 3.8 or higher
- Google Cloud API Key with Gemini API access
- Set environment variable:
  ```bash
  export GEMINI_API_KEY=your_api_key_here
  ```

## Usage

1. Set up Python virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Process dataset
```bash
python emotion_dataset_to_text_embedding.py
python text_embedding_to_fine_tuning.py
```

4. Create and run model fine-tuning
```bash
python create_tuned_model.py
```

5. Manage fine-tuned models
```bash
# List all tuned models
python manage_tuned_models.py --list

# Check status of a specific model
python manage_tuned_models.py --status "models/your-model-name"

# Delete a specific model
python manage_tuned_models.py --delete "models/your-model-name"
```

## Model Fine-tuning Configuration

The fine-tuning process uses the following default parameters:
- Base Model: `models/gemini-1.5-flash-001-tuning`
- Batch Size: 4
- Learning Rate: 0.001
- Target Epochs: 100 (automatically adjusted based on dataset size)
- Maximum Total Examples: 250,000

These parameters can be modified in `create_tuned_model.py` if needed.


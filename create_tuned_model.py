import google.genai as genai
from google.genai import types
import os
import json
import time
import uuid
import random
import math

# --- Configuration ---
BASE_MODEL_ID = "models/gemini-1.5-flash-001-tuning" 

# Set the ID and display name for the fine-tuned model to be created.
# The ID should be unique; consider adding a UUID if necessary.
# This is the name that will be displayed in Google AI Studio.
tuned_model_display_name = f"ondaum-gemini-1.5-flash-{time.strftime('%Y%m%dT%H%M%S')}" # User-defined

# Unique ID for the fine-tuned model. Use lowercase English letters, numbers, and hyphens (-).
# To ensure uniqueness, you can generate a unique ID each time the script runs or specify one manually.
# Example: f"tuned-model-{int(time.time())}"
tuned_model_id = f"ondaum-tuned-model-{str(uuid.uuid4())[:8]}"


TRAINING_DATA_PATH = "./dataset/finetune_dataset.jsonl" # Path to your training dataset file (JSONL format)

# Hyperparameters (adjust as needed)
MAX_TOTAL_EXAMPLES = 250000  # Maximum allowed examples * epochs
TARGET_EPOCHS = 100  # Target number of epochs
BATCH_SIZE = 4    # Batch size (default from tutorial)
LEARNING_RATE = 0.001 # Learning rate (default from tutorial)

# Set API Key (retrieve from environment variable)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Run 'export GEMINI_API_KEY=YOUR_API_KEY'.")

# Initialize the client
client = genai.Client(api_key=GEMINI_API_KEY)

def calculate_epoch_count(data_size):
    """
    Calculate the appropriate epoch count based on data size and MAX_TOTAL_EXAMPLES.
    """
    max_epochs = MAX_TOTAL_EXAMPLES // data_size
    return min(TARGET_EPOCHS, max_epochs)

def sample_data(data, target_size):
    """
    Randomly sample data to reach target size while maintaining distribution.
    """
    if len(data) <= target_size:
        return data
    
    # Shuffle the data to ensure random sampling
    random.shuffle(data)
    return data[:target_size]

def load_and_prepare_data(file_path):
    """
    Loads a JSONL file and converts it to the Gemini API format.
    Maps custom keys 'input_text', 'output_text' to API standard 'text_input', 'output'.
    """
    print(f"Loading and preparing data: {file_path}")
    prepared_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    original_record = json.loads(line)
                    # Map user keys to API standard keys
                    # User dataset keys: "input_text", "output_text"
                    # Gemini API expected keys: "text_input", "output"
                    if "input_text" not in original_record or "output_text" not in original_record:
                        print(f"Warning: Line {i+1} is missing 'input_text' or 'output_text' key. Skipping.")
                        continue

                    prepared_record = {
                        "text_input": original_record["input_text"],
                        "output": original_record["output_text"]
                    }
                    prepared_data.append(prepared_record)
                except json.JSONDecodeError:
                    print(f"Warning: Line {i+1} is not valid JSON. Skipping.")
                except KeyError as e:
                    print(f"Warning: Line {i+1} is missing a required key ({e}). Skipping: {line.strip()}")

        # Calculate maximum allowed data size based on target epochs
        max_data_size = MAX_TOTAL_EXAMPLES // TARGET_EPOCHS
        
        if len(prepared_data) > max_data_size:
            print(f"Data size ({len(prepared_data)}) exceeds maximum allowed size ({max_data_size}) for {TARGET_EPOCHS} epochs.")
            print("Sampling data to meet the requirements...")
            prepared_data = sample_data(prepared_data, max_data_size)
            print(f"Sampled data size: {len(prepared_data)}")

        # Calculate appropriate epoch count
        epoch_count = calculate_epoch_count(len(prepared_data))
        print(f"Adjusted epoch count: {epoch_count} (based on data size: {len(prepared_data)})")

        # Gemini API might require a minimum of 20 examples. (Tutorials often recommend 100-500)
        if len(prepared_data) < 20:
            print(f"Warning: The number of prepared data examples ({len(prepared_data)}) is very low. Fine-tuning may fail or perform poorly.")
        elif len(prepared_data) < 100:
            print(f"Info: The number of prepared data examples is {len(prepared_data)}. More data is recommended for better results.")
        else:
            print(f"Data loading complete. Total {len(prepared_data)} examples prepared.")
        return prepared_data, epoch_count
    except FileNotFoundError:
        print(f"Error: Training data file ({file_path}) not found.")
        raise
    except Exception as e:
        print(f"Error during data loading: {e}")
        raise

def main():
    """
    Main function to execute the fine-tuning process.
    """
    print("Gemini Model Fine-Tuning Script Started")
    print("------------------------------------")
    print(f"Base Model: {BASE_MODEL_ID}")
    print(f"Training Data: {TRAINING_DATA_PATH}")
    print(f"Tuned Model ID to be created: {tuned_model_id}")
    print(f"Tuned Model Display Name: {tuned_model_display_name}")
    print(f"Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
    print("------------------------------------\n")

    # 1. Load and prepare data
    training_data, epoch_count = load_and_prepare_data(TRAINING_DATA_PATH)
    if not training_data:
        print("No training data available. Exiting script.")
        return

    # 2. Request fine-tuned model creation
    print(f"\nStarting fine-tuning job based on model '{BASE_MODEL_ID}'...")
    try:
        # Create tuning job using the new client API
        tuning_job = client.tunings.tune(
            base_model=BASE_MODEL_ID,
            training_dataset=types.TuningDataset(
                examples=[
                    types.TuningExample(
                        text_input=example["text_input"],
                        output=example["output"]
                    )
                    for example in training_data
                ]
            ),
            config=types.CreateTuningJobConfig(
                epoch_count=epoch_count,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                tuned_model_display_name=tuned_model_display_name,
            )
        )

        print(f"Fine-tuning job initiated. Job Name: {tuning_job.name}")
        print(f"Current State: {tuning_job.state}")

        # 3. Monitor tuning status
        print("\nWaiting for tuning to complete... (Checking status periodically)")
        running_states = {"JOB_STATE_QUEUED", "JOB_STATE_PENDING", "JOB_STATE_RUNNING"}
        
        while tuning_job.state in running_states:
            print(f" - Current State: {tuning_job.state} (Checked at: {time.strftime('%Y-%m-%d %H:%M:%S')})")
            time.sleep(60)
            try:
                tuning_job = client.tunings.get(name=tuning_job.name)
            except Exception as e:
                print(f"Error while checking status: {e}. Retrying after a delay.")
                time.sleep(60)

        # 4. Completion and result check
        if tuning_job.state == "JOB_STATE_SUCCEEDED":
            print("\n🎉 Fine-tuning completed successfully! 🎉")
            print(f"Job Name: {tuning_job.name}")
            print(f"Display Name: {tuning_job.display_name}")
            print(f"Description: {tuning_job.description}")
            print(f"Base Model: {tuning_job.base_model}")
            print(f"State: {tuning_job.state}")
            print(f"Creation Time: {tuning_job.create_time}")
            print(f"Last Update Time: {tuning_job.update_time}")

            # Get the tuned model
            tuned_model = client.models.get(model=tuning_job.tuned_model.model)
            print(f"\nTuned Model Name: {tuned_model.name}")
            print(f"Tuned Model Display Name: {tuned_model.display_name}")

        elif tuning_job.state == "JOB_STATE_FAILED":
            print("\n❌ Fine-tuning job failed.")
            print(f"State: {tuning_job.state}")
            if hasattr(tuning_job, 'error') and tuning_job.error:
                print(f"Error Message: {tuning_job.error.message}")
            else:
                print("Check Google Cloud Console or API response for detailed error information.")
        else:
            print(f"\nFine-tuning job ended with an unexpected state: {tuning_job.state}")

    except Exception as e:
        print(f"A critical error occurred during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
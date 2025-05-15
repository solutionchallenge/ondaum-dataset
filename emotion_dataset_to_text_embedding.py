# Imports
import google.generativeai as genai
import pandas as pd
import json
import time
import os
import sys
import re
import threading # For RateLimiter lock
import concurrent.futures # Although not used in final sequential batch, good practice if concurrency added later
from dotenv import load_dotenv # Optional

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

CLASSIFICATION_MODEL_NAME = "gemini-1.5-pro-latest"
EMBEDDING_MODEL_NAME = "text-embedding-004"
INPUT_CSV_PATH = "./dataset/korean_emotion_dataset.csv"
OUTPUT_JSONL_PATH = "./dataset/reclassified_dataset.jsonl" # Adjusted output name
SENTENCE_COLUMN = "Sentence"
KOREAN_LABEL_COLUMN = "Emotion"
EMOTION_CATEGORIES = ["Joy", "Sadness", "Anger", "Surprise", "Fear", "Disgust", "Neutral"]
BATCH_SIZE = 12 # Process in batches of 5
API_RETRY_LIMIT = 3
API_RETRY_DELAY = 5
# Adjust Rate Limits based on your API plan/quotas (requests per second)
CLASSIFICATION_RPM = 60 # Example: 60 requests per minute
EMBEDDING_RPM = 300    # Example: 300 requests per minute
CLASSIFICATION_RPS = CLASSIFICATION_RPM / 60.0
EMBEDDING_RPS = EMBEDDING_RPM / 60.0 # Note: Batch embedding is one call per BATCH_SIZE items

CLASSIFICATION_PROMPT_TEMPLATE = f"""
주어진 문장의 감정을 다음 영어 카테고리 중 하나로 분류해주세요: {', '.join(EMOTION_CATEGORIES)}.

참고용 기존 한국어 레이블: '{{original_korean_label}}'

문장 내용을 분석하여 가장 적절한 **영어** 감정 레이블 하나와, 그 판단에 대한 **확신도 점수(0.0에서 1.0 사이의 숫자)**를 아래 형식에 맞춰 답변해주세요. 다른 설명은 필요 없습니다.

문장: "{{sentence_text}}"

형식:
Emotion: [영어 감정 레이블]
Confidence: [확신도 점수 (숫자)]

결과:
"""

# --- Rate Limiter Class ---
class RateLimiter:
    """A simple token bucket rate limiter."""
    def __init__(self, rate, per_seconds=1):
        self.rate = rate / per_seconds # Tokens per second
        self.capacity = float(rate)    # Bucket capacity
        self.tokens = self.capacity    # Start with a full bucket
        self.last_time = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_time
            self.last_time = now

            # Add tokens generated during elapsed time
            self.tokens += elapsed * self.rate
            if self.tokens > self.capacity:
                self.tokens = self.capacity

            # If not enough tokens, wait
            if self.tokens < 1.0:
                required_wait = (1.0 - self.tokens) / self.rate
                # print(f"Rate limit hit, waiting for {required_wait:.2f} seconds...") # Optional debug log
                time.sleep(required_wait)
                self.tokens = 0 # Waited, so bucket is now empty until next calculation
            else:
                self.tokens -= 1.0 # Consume one token

# --- Helper Functions ---

def configure_gemini():
    """Configures the Gemini API client."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Error configuring Gemini API: {e}")
        sys.exit(1)

def parse_classification_response(response_text: str) -> tuple[str | None, float | None]:
    """Parses the Gemini response to extract emotion label and confidence score."""
    emotion = None
    confidence = None
    try:
        emotion_match = re.search(r"Emotion:\s*(\w+)", response_text, re.IGNORECASE | re.UNICODE)
        if emotion_match:
            potential_emotion = emotion_match.group(1).strip()
            for cat in EMOTION_CATEGORIES:
                if potential_emotion.lower() == cat.lower():
                    emotion = cat
                    break
        confidence_match = re.search(r"Confidence:\s*([0-9.]+)", response_text, re.IGNORECASE | re.UNICODE)
        if confidence_match:
            try:
                potential_confidence = float(confidence_match.group(1).strip())
                if 0.0 <= potential_confidence <= 1.0:
                    confidence = potential_confidence
            except ValueError: pass # Keep None if conversion fails
    except Exception as e:
        print(f"  Error parsing response text: {e}\nResponse Text:\n{response_text[:200]}...") # Log snippet
    return emotion, confidence

def get_gemini_classification(text: str, original_label: str, model: genai.GenerativeModel, limiter: RateLimiter, retries=API_RETRY_LIMIT) -> tuple[str | None, float | None]:
    """Uses Gemini Pro model to classify the text emotion, with rate limiting."""
    if not text or pd.isna(text): return None, None
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
        sentence_text=text,
        original_korean_label=original_label if original_label and not pd.isna(original_label) else "N/A"
    )
    attempt = 0
    while attempt < retries:
        try:
            limiter.wait() # Wait before making the API call
            # print(f"  Calling classification for: '{text[:30]}...'") # Optional debug log
            response = model.generate_content(prompt)
            # print(f"  Received classification response.") # Optional debug log
            predicted_label, confidence_score = parse_classification_response(response.text)
            if predicted_label is not None:
                 return predicted_label, confidence_score
            else: # Parsing failed, treat as retryable error for now
                 print(f"  Classification parsing failed for text: '{text[:50]}...'. Retrying ({attempt+1}/{retries})")
                 attempt += 1
                 if attempt < retries: time.sleep(API_RETRY_DELAY)
                 else: print(f"  Max retries reached for parsing failure: '{text[:50]}...'"); return None, None
        except Exception as e:
            attempt += 1
            print(f"Error during classification API call (Attempt {attempt}/{retries}): {e}")
            if attempt < retries: time.sleep(API_RETRY_DELAY)
            else: print("Max retries reached for classification."); return None, None
    return None, None

def get_gemini_embeddings_batch(texts: list[str], limiter: RateLimiter, model_name: str = EMBEDDING_MODEL_NAME, retries=API_RETRY_LIMIT) -> list[list[float] | None]:
    """Gets text embeddings for a batch of texts, with rate limiting."""
    if not texts: return []
    valid_texts = []
    original_indices_map = []
    for i, text in enumerate(texts):
        if text and not pd.isna(text):
            valid_texts.append(text)
            original_indices_map.append(i)

    if not valid_texts: return [None] * len(texts)

    final_embeddings = [None] * len(texts)
    attempt = 0
    while attempt < retries:
        try:
            limiter.wait() # Wait once before the batch API call
            # print(f"  Calling batch embedding for {len(valid_texts)} sentences...") # Optional debug log
            result = genai.embed_content(model=f"models/{model_name}",
                                         content=valid_texts,
                                         task_type="RETRIEVAL_DOCUMENT")
            # print(f"  Received batch embedding response.") # Optional debug log
            embeddings_for_valid_texts = result.get('embedding')

            if embeddings_for_valid_texts and len(embeddings_for_valid_texts) == len(valid_texts):
                # Reconstruct the full list including Nones for failed/empty inputs
                for i, emb in enumerate(embeddings_for_valid_texts):
                    original_index = original_indices_map[i]
                    final_embeddings[original_index] = emb if emb else None
                return final_embeddings # Success
            else:
                print(f"Warning: Mismatch in embedding count or no embeddings found for batch size {len(valid_texts)}. Retrying ({attempt+1}/{retries}).")
                attempt += 1
                if attempt < retries: time.sleep(API_RETRY_DELAY)
                else: print("Max retries reached for batch embedding size mismatch."); return [None] * len(texts)

        except Exception as e:
            attempt += 1
            print(f"Error during batch embedding API call (Attempt {attempt}/{retries}): {e}")
            if attempt < retries: time.sleep(API_RETRY_DELAY)
            else: print("Max retries reached for batch embedding."); return [None] * len(texts)
    return [None] * len(texts) # Should not be reached if retry logic is correct

# --- Main Execution ---
if __name__ == "__main__":
    configure_gemini()
    # Initialize models (consider doing this inside main if preferred)
    try:
        classification_model = genai.GenerativeModel(CLASSIFICATION_MODEL_NAME)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize classification model '{CLASSIFICATION_MODEL_NAME}': {e}")
        sys.exit(1)
    # Embedding model is called via genai.embed_content, no separate init needed here

    # --- Load CSV ---
    print(f"Reading input CSV file: {INPUT_CSV_PATH}")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Found {len(df)} total rows in the CSV.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Input CSV file not found at {INPUT_CSV_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Error reading CSV: {e}")
        sys.exit(1)

    # --- Validate CSV Columns ---
    if SENTENCE_COLUMN not in df.columns or KOREAN_LABEL_COLUMN not in df.columns:
         print(f"CRITICAL ERROR: CSV must contain columns named '{SENTENCE_COLUMN}' and '{KOREAN_LABEL_COLUMN}'")
         sys.exit(1)

    # --- Resumability: Load Processed Indices ---
    processed_indices = set()
    if os.path.exists(OUTPUT_JSONL_PATH):
        print(f"Output file '{OUTPUT_JSONL_PATH}' found. Loading previously processed row indices...")
        try:
            with open(OUTPUT_JSONL_PATH, 'r', encoding='utf-8') as infile:
                processed_lines_count = 0
                for line in infile:
                    try:
                        data = json.loads(line)
                        if 'original_index' in data and isinstance(data['original_index'], int):
                            processed_indices.add(data['original_index'])
                            processed_lines_count += 1
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping corrupted JSON line in existing output file: {line.strip()}")
                print(f"Found {len(processed_indices)} unique processed row indices in {processed_lines_count} lines.")
        except Exception as e:
            print(f"Warning: Error reading existing output file: {e}. Proceeding, but results might be duplicated if writing occurred before.")
            # Keep existing processed_indices if partially read, or empty if read failed early
    else:
        print("No existing output file found. Starting fresh.")

    # --- Batch Processing Logic ---
    batch_to_process = [] # Holds dicts: {'index': i, 'sentence': s, 'original_label': l}
    processed_count_this_run = 0
    error_count_this_run = 0
    skipped_count = 0
    total_rows_checked = 0

    # Initialize Rate Limiters
    # Note: Use requests per second for the RateLimiter class
    classification_limiter = RateLimiter(CLASSIFICATION_RPS, 1)
    # Embedding limiter applies to the BATCH call, so effectively allows BATCH_SIZE embeddings per second / EMBEDDING_RPS
    # The limiter controls the rate of the BATCH calls themselves.
    embedding_limiter = RateLimiter(EMBEDDING_RPS, 1) # This limits batch calls, not individual embeddings within

    print(f"\nStarting/Resuming processing in batches of {BATCH_SIZE}. Appending results to: {OUTPUT_JSONL_PATH}")
    print(f"Rate limits: Classification ~{CLASSIFICATION_RPM} RPM, Embedding Batches ~{EMBEDDING_RPM} RPM")

    # --- Main Loop ---
    for index, row in df.iterrows():
        total_rows_checked += 1
        # --- Check if already processed ---
        if index in processed_indices:
            skipped_count += 1
            continue # Skip silently or log less frequently

        # --- Add to batch if not processed ---
        sentence = str(row[SENTENCE_COLUMN]).strip()
        original_label = str(row[KOREAN_LABEL_COLUMN]).strip()

        if not sentence: # Skip rows with empty sentences
            print(f"Warning: Skipping row {index + 1} due to empty sentence.")
            continue

        batch_to_process.append({
            'index': index,
            'sentence': sentence,
            'original_label': original_label
        })

        # --- Process batch when full or at the end of DataFrame ---
        # Process if batch is full OR it's the last row of the DataFrame
        is_last_row = (total_rows_checked == len(df))
        if len(batch_to_process) == BATCH_SIZE or (is_last_row and batch_to_process):
            current_batch = batch_to_process[:] # Copy the batch
            batch_to_process = [] # Clear for next batch immediately
            batch_start_index = current_batch[0]['index']
            print(f"\nProcessing batch of {len(current_batch)} (starting index {batch_start_index})...")

            batch_results_data = [] # Holds the final dicts to be written for this batch

            # 1. Batch Embedding
            batch_sentences = [item['sentence'] for item in current_batch]
            # Apply limiter for the single batch embedding call
            # embedding_limiter.wait() - Moved inside the function for clarity
            batch_embeddings = get_gemini_embeddings_batch(batch_sentences, embedding_limiter)

            # 2. Sequential Classification and Result Assembly
            successful_items_in_batch = 0
            for i, item in enumerate(current_batch):
                current_index = item['index']
                current_sentence = item['sentence']
                current_original_label = item['original_label']
                embedding_vector = batch_embeddings[i] # Get corresponding embedding

                if embedding_vector is None:
                    print(f"  -> Skipping index {current_index} (in batch starting {batch_start_index}): Embedding failed.")
                    error_count_this_run +=1
                    continue # Skip this item if embedding failed

                # Call classification individually, applying its limiter
                # limiter is passed to the function which calls wait()
                predicted_emotion, confidence = get_gemini_classification(
                    current_sentence, current_original_label, classification_model, classification_limiter
                )

                if not predicted_emotion:
                    print(f"  -> Skipping index {current_index} (in batch starting {batch_start_index}): Classification failed.")
                    error_count_this_run +=1
                    continue # Skip this item if classification failed

                # If both succeeded, prepare data for writing
                result_data = {
                    "original_index": current_index,
                    "sentence": current_sentence,
                    "emotion": predicted_emotion,
                    "confidence_score": confidence,
                    "embedding_vector": embedding_vector
                }
                batch_results_data.append(result_data)
                successful_items_in_batch += 1

            # 3. Write successful results for the batch (if any)
            if batch_results_data:
                try:
                    # Open in append mode for each batch write to ensure data is saved
                    with open(OUTPUT_JSONL_PATH, 'a', encoding='utf-8') as outfile:
                        for data_to_write in batch_results_data:
                            json_line = json.dumps(data_to_write, ensure_ascii=False)
                            outfile.write(json_line + '\n')
                        processed_count_this_run += len(batch_results_data)
                        # Add successfully written indices to the set immediately
                        for written_data in batch_results_data:
                            processed_indices.add(written_data['original_index'])
                        print(f"  Batch starting {batch_start_index}: Successfully processed and wrote {len(batch_results_data)} items.")
                except Exception as e:
                    print(f"CRITICAL ERROR: Failed to write batch results (starting index {batch_start_index}) to file: {e}")
                    # If write fails, items are NOT added to processed_indices, so they will be retried.
                    # Consider adding failed items to a separate error log.
                    error_count_this_run += len(batch_results_data) # Count write failures

            # Optional: Log progress periodically
            if total_rows_checked % (BATCH_SIZE * 10) == 0: # Log every 10 batches
                 print(f"Progress: Checked ~{total_rows_checked}/{len(df)} rows. Successful this run: {processed_count_this_run}. Errors this run: {error_count_this_run}")


    # --- Final Summary ---
    print("\n--- Final Processing Summary ---")
    print(f"Total rows in CSV: {len(df)}")
    # Recalculate accurately from file if possible, otherwise use in-memory count
    final_processed_count = len(processed_indices)
    print(f"Rows successfully processed/written across all runs (unique indices): {final_processed_count}")
    print(f"Rows skipped (already processed in previous runs): {skipped_count}")
    print(f"Rows successfully processed in the last run: {processed_count_this_run}")
    print(f"Rows failed or skipped due to errors in the last run (will retry next time): {error_count_this_run}")
    if final_processed_count + error_count_this_run + skipped_count < len(df):
         print(f"Note: Some rows may not have been attempted if script stopped prematurely.")
    print(f"Output is in: {OUTPUT_JSONL_PATH}")
    print("Script finished.")
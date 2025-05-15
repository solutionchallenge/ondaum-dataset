import google.genai as genai
import os
import argparse
from datetime import datetime
import time

def setup_client():
    """Initialize Gemini API client with API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)

def list_tuned_models(client):
    """List all tuned models associated with the API key."""
    try:
        models = client.models.list()
        tuned_models = [model for model in models if "tuned" in model.name]
        
        if not tuned_models:
            print("No tuned models found.")
            return
        
        print("\n=== Tuned Models ===")
        for model in tuned_models:
            print(f"\nModel Name: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Creation Time: {model.create_time}")
            print(f"Last Update Time: {model.update_time}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error listing models: {e}")

def get_model_status(client, model_name):
    """Get detailed status of a specific tuned model."""
    try:
        model = client.models.get(model=model_name)
        print(f"\n=== Model Status: {model.display_name} ===")
        print(f"Model Name: {model.name}")
        print(f"Display Name: {model.display_name}")
        print(f"Creation Time: {model.create_time}")
        print(f"Last Update Time: {model.update_time}")
        print(f"State: {model.state}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error getting model status: {e}")

def delete_tuned_model(client, model_name):
    """Delete a specific tuned model."""
    try:
        print(f"\nAttempting to delete model: {model_name}")
        client.models.delete(model=model_name)
        print(f"Successfully deleted model: {model_name}")
        
    except Exception as e:
        print(f"Error deleting model: {e}")

def main():
    parser = argparse.ArgumentParser(description='Manage Gemini tuned models')
    parser.add_argument('--list', action='store_true', help='List all tuned models')
    parser.add_argument('--status', type=str, help='Get status of a specific model')
    parser.add_argument('--delete', type=str, help='Delete a specific model')
    
    args = parser.parse_args()
    
    try:
        client = setup_client()
        
        if args.list:
            list_tuned_models(client)
        elif args.status:
            get_model_status(client, args.status)
        elif args.delete:
            confirm = input(f"Are you sure you want to delete model {args.delete}? (yes/no): ")
            if confirm.lower() == 'yes':
                delete_tuned_model(client, args.delete)
            else:
                print("Deletion cancelled.")
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 
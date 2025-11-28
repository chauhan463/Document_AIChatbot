import time
from llama_cpp import Llama
import os

# --- THIS IS THE PATH WE CREATED ---
model_path = os.path.join(os.getcwd(), "ai-models", "Meta-Llama-3-8B-Instruct.Q5_K_M.gguf")
# ----------------------------------------

print(f"Loading model from: {model_path}")
if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at that path!")
    print("Please check the path and filename.")
    exit()

# Initialize the Llama model
# n_gpu_layers=-1 tells it to offload all possible layers to the GPU
try:
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,  # Offload all layers to GPU
        n_ctx=2048,       # Context window size
        verbose=True      # Show detailed loading output
    )
except Exception as e:
    print(f"ERROR: Failed to load model. {e}")
    exit()

print("\nModel loaded successfully! Running a test...")
start_time = time.time()

# Define the messages for the chat
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Hello! In one sentence, who are you?"
    }
]

# Generate a response
try:
    response = llm.create_chat_completion(
        messages=messages
    )

    end_time = time.time()

    # Print the result
    if response and 'choices' in response and len(response['choices']) > 0 and 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
        answer = response['choices'][0]['message']['content']
        print("\n--- Test Result ---")
        print(f"LLM Response: {answer}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print("-------------------")

        if "assistant" in answer.lower():
            print("\nSUCCESS! The LLM is loaded on the GPU and responding.")
        else:
            print("\nTest complete. The model generated a response.")
    else:
        print("\nERROR: Received an unexpected response format from the model.")
        print(f"Raw Response: {response}")


except Exception as e:
    print(f"ERROR: Failed to generate response. {e}")

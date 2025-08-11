import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Set the device to use (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained CLIP model and processor from Hugging Face
# This model is great for general-purpose image-text matching.
MODEL_NAME = "openai/clip-vit-base-patch32"

try:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print("CLIP model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have an internet connection and the 'transformers' library is installed.")
    # Exit or handle the error appropriately if the model can't be loaded
    exit()

def calculate_clip_score(image_path: str, text: str) -> float:
    """
    Calculates the CLIP score to measure the similarity between an image and a text prompt.

    Args:
        image_path (str): The file path to the image.
        text (str): The text prompt to compare against the image.

    Returns:
        float: The calculated CLIP score, scaled to a range of 0-100.
               Returns 0.0 if the image cannot be found.
    """
    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return 0.0

    # Use the processor to prepare the image and text for the model
    inputs = processor(
        text=[text], 
        images=image, 
        return_tensors="pt", # Return PyTorch tensors
        padding=True
    ).to(device)

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Generate image and text embeddings
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
        # Normalize the features to have unit length
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate the cosine similarity and scale it by 100
        # The dot product of unit vectors is the cosine similarity
        similarity_score = (text_features @ image_features.T).item() * 100.0

    return similarity_score

# --- Example Usage ---

# IMPORTANT: Replace 'path/to/your/image.jpg' with the actual path to your image file.
# For example: 'C:/Users/YourUser/Pictures/my_dog.jpg' or './images/cat.png'
image_file = 'path/to/your/image.jpg'

# 1. A text prompt that closely matches the image content
prompt_good_match = "a photograph of a golden retriever playing in a park"
score1 = calculate_clip_score(image_file, prompt_good_match)

if score1 > 0: # Only print if the image was found
    print(f"\nImage: '{image_file}'")
    print(f"Prompt: '{prompt_good_match}'")
    print(f"✅ CLIP Score: {score1:.2f}")

# 2. A text prompt that does not match the image content
prompt_bad_match = "a drawing of a spaceship landing on the moon"
score2 = calculate_clip_score(image_file, prompt_bad_match)

if score2 > 0:
    print(f"\nImage: '{image_file}'")
    print(f"Prompt: '{prompt_bad_match}'")
    print(f"❌ CLIP Score: {score2:.2f}")


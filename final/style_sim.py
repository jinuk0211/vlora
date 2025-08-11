import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Set the device to use (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained VGG-19 model
# We use the 'features' part of the model, which contains the convolutional layers.
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# Define the layers from which we'll extract style features
STYLE_LAYERS = {
    '0': 'conv1_1',
    '5': 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '28': 'conv5_1'
}

def get_features(image, model, layers):
    """Extracts features from the specified layers of a model."""
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """
    Calculates the Gram matrix of a given tensor.
    The Gram matrix captures the correlations between feature maps, representing style.
    """
    # Get the dimensions of the tensor
    b, c, h, w = tensor.size()
    
    # Reshape the tensor to combine height and width into a single dimension
    features = tensor.view(b, c, h * w)
    
    # Compute the Gram product
    gram = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by dividing by the number of elements
    return gram.div(c * h * w)

def get_image_tensor(image_path):
    """Loads and preprocesses an image, returning it as a tensor."""
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None

    # Define the transformation pipeline for the image
    # It must match the preprocessing used for the VGG-19 model.
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply the transformations and add a batch dimension
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor.to(device)

def calculate_style_similarity(image1_path: str, image2_path: str) -> float:
    """
    Calculates the style similarity score between two images.

    Args:
        image1_path (str): The file path to the first image.
        image2_path (str): The file path to the second image.

    Returns:
        float: A similarity score between 0.0 and 1.0. 
               A score of 1.0 means identical styles.
    """
    # Load and preprocess the images
    img1_tensor = get_image_tensor(image1_path)
    img2_tensor = get_image_tensor(image2_path)

    if img1_tensor is None or img2_tensor is None:
        return 0.0

    # Extract features for both images
    img1_features = get_features(img1_tensor, vgg, STYLE_LAYERS)
    img2_features = get_features(img2_tensor, vgg, STYLE_LAYERS)

    # Calculate the style loss
    style_loss = 0
    for layer in STYLE_LAYERS.values():
        # Calculate Gram matrices for the features from each image
        gram1 = gram_matrix(img1_features[layer])
        gram2 = gram_matrix(img2_features[layer])
        
        # Add the mean squared error between the Gram matrices to the total loss
        style_loss += nn.functional.mse_loss(gram1, gram2)

    # Convert the loss (a measure of difference) into a similarity score
    # The formula 1 / (1 + loss) maps a loss of 0 to a similarity of 1,
    # and larger losses to similarities approaching 0.
    similarity_score = 1 / (1 + style_loss.item())
    
    return similarity_score

# --- Example Usage ---

# Use the penguin image you uploaded and find another image to compare it with.
# For example, a painting by Van Gogh.
penguin_image = "penguin two.jpeg"
# IMPORTANT: Replace 'path/to/van_gogh.jpg' with an actual file path.
vangogh_image = "path/to/van_gogh.jpg" 

# 1. Compare the penguin photo to a classic painting (low similarity expected)
score1 = calculate_style_similarity(penguin_image, vangogh_image)
if score1 > 0:
    print(f"Style similarity between '{penguin_image}' and '{vangogh_image}':")
    print(f"🎨 Score: {score1:.4f}")

# 2. Compare the penguin photo to itself (perfect similarity expected)
score2 = calculate_style_similarity(penguin_image, penguin_image)
if score2 > 0:
    print(f"\nStyle similarity between '{penguin_image}' and itself:")
    print(f"🐧 Score: {score2:.4f}")
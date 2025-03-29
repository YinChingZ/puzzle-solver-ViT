import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.puzzle_solver import PuzzleSolver

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def display_image(image_tensor):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    # Load the pre-trained model
    model_path = 'path/to/pretrained/model.pth'
    model = PuzzleSolver()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load and preprocess the image
    image_path = 'path/to/demo/image.jpg'
    image = load_image(image_path, transform)
    image = image.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        position_logits, relation_logits, reconstructed_image = model(image)

    # Display the reconstructed image
    display_image(reconstructed_image.squeeze(0))

if __name__ == '__main__':
    main()

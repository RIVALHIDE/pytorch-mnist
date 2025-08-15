import torch
from PIL import Image
from torch import nn, save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

# Device setup (use GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load MNIST dataset
train = datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=ToTensor()
)

# DataLoader for batching
dataset = DataLoader(train, batch_size=32, shuffle=True)

# Define the CNN model
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10)  # Output: 10 classes
        )

    def forward(self, x):
        return self.model(x)

# Create model instance
clf = ImageClassifier().to(device)

# Print parameter count for debugging
print(f"Total parameters: {sum(p.numel() for p in clf.parameters())}")

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
opt = Adam(clf.parameters(), lr=1e-3)

# Training loop
if __name__ == "__main__":
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f, map_location=device))
    clf.eval()  # Set model to evaluation mode

    # Image preprocessing
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
    ])


    img = Image.open('img_1.jpg')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = clf(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
        print(f"Predicted digit: {prediction}")

    print(torch.argmax(clf(img_tensor)))


    # for epoch in range(10):  # Train for 10 epochs
    #     for batch in dataset:
    #         X, y = batch
    #         X, y = X.to(device), y.to(device)

    #         # Forward pass
    #         yhat = clf(X)
    #         loss = loss_fn(yhat, y)

    #         # Backward pass
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #     print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

    # # Save the model state
    # save(clf.state_dict(), 'model_state.pt')
    # print("Model saved to model_state.pt")

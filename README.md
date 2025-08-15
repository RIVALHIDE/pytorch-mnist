🖤 MNIST Digit Classifier – PyTorch  

A Convolutional Neural Network (CNN)** built using PyTorch to classify handwritten digits from the MNIST dataset.  
This project loads a trained model and predicts digits from custom images.  

---

📌 Features  
- Dataset: MNIST (28×28 grayscale digit images)  
- Model Architecture:  
  - 3 Convolutional Layers + ReLU activation  
  - Flatten layer  
  - Fully Connected Layer (10 outputs for digits 0–9)  
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  
- Device Support: GPU (CUDA) or CPU  

---

📂 Project Structure 
├── data/ # MNIST dataset (auto-downloaded)
├── img_1.jpg # Sample image for prediction
├── model_state.pt # Saved model weights
├── torchnn.py # Main script
└── README.md # Documentation

---

 🚀 Installation  

1️⃣ Clone the repository :
--bash :
git clone https://github.com/RIVALHIDE/pytorch-minst.git
cd RIVALHIDE

2️⃣ Create a virtual environment :
--bash:
python -m venv .venv

3️⃣ Activate the environment :

Windows:
--bash :
.venv\Scripts\activate

4️⃣ Install dependencies :
--bash :
pip install torch torchvision pillow

▶️ Usage :

Run the prediction script:
--bash :
python torchnn.py

Example Output:
Using device: cpu
Total parameters: 365514
Predicted Digit: 7


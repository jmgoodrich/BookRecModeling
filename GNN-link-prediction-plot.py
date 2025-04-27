import pickle
import matplotlib.pyplot as plt

# === Load training history ===
history_path = "C:\\Users\\fahim\\Documents\\Courses\\AI Modeling\\training_history.pkl"

with open(history_path, "rb") as f:
    history = pickle.load(f)

epochs = history['epochs']
losses = history['losses']
aucs = history['aucs']
aps = history['aps']

# === Plot Loss ===
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', label='Loss', color='tab:red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Plot AUC and AP ===
plt.figure(figsize=(10, 6))
plt.plot(epochs, aucs, marker='o', label='AUC', color='tab:blue')
plt.plot(epochs, aps, marker='s', label='AP', color='tab:green')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('AUC and AP over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

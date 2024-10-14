import matplotlib.pyplot as plt

# Data for training loss and validation accuracy over 2 epochs
epochs = [1, 2]
train_loss = [1.0714, 1.0471]  # Loss values for illustration
valid_acc = [0.4425, 0.44125]  # Validation accuracy for RNN (Hidden Dim = 64)

# Plotting the learning curve
plt.figure(figsize=(10, 5))

# Training Loss curve
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

# Validation Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(epochs, valid_acc, label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()

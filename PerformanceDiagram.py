import matplotlib.pyplot as plt

# FFNN data
ffnn_hidden_dims = [10, 50]
ffnn_train_acc = [0.513, 0.527]
ffnn_val_acc = [0.55875, 0.535]

# RNN data (epoch 1 and epoch 2)
rnn_hidden_dims = [32, 64]
rnn_train_acc_epoch1 = [0.41875, 0.4055]
rnn_val_acc_epoch1 = [0.43375, 0.4425]
rnn_train_acc_epoch2 = [0.437125, 0.426375]
rnn_val_acc_epoch2 = [0.42125, 0.44125]

plt.figure(figsize=(12, 6))

# Plot FFNN
plt.subplot(1, 2, 1)
plt.plot(ffnn_hidden_dims, ffnn_train_acc, label='FFNN Training Accuracy', marker='o')
plt.plot(ffnn_hidden_dims, ffnn_val_acc, label='FFNN Validation Accuracy', marker='o')
plt.xlabel('Hidden Dimension')
plt.ylabel('Accuracy')
plt.title('FFNN Accuracy vs Hidden Dimension')
plt.legend()

# Plot RNN
plt.subplot(1, 2, 2)
plt.plot(rnn_hidden_dims, rnn_train_acc_epoch1, label='RNN Training Accuracy (Epoch 1)', marker='o')
plt.plot(rnn_hidden_dims, rnn_val_acc_epoch1, label='RNN Validation Accuracy (Epoch 1)', marker='o')
plt.plot(rnn_hidden_dims, rnn_train_acc_epoch2, label='RNN Training Accuracy (Epoch 2)', marker='o')
plt.plot(rnn_hidden_dims, rnn_val_acc_epoch2, label='RNN Validation Accuracy (Epoch 2)', marker='o')
plt.xlabel('Hidden Dimension')
plt.ylabel('Accuracy')
plt.title('RNN Accuracy vs Hidden Dimension')
plt.legend()

plt.tight_layout()
plt.show()
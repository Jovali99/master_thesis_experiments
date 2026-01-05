import matplotlib.pyplot as plt

class VisualizeModel():
    """Class used for visualization of a trained models evaluation"""

    def visualize(self, train_result, test_result):
        train_acc = train_result.metrics.extra["accuracy_history"]
        train_loss = train_result.metrics.extra["loss_history"]
        test_acc = test_result.accuracy
        test_loss = test_result.loss

        # Plot training and test accuracy
        plt.figure(figsize=(5, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(len(train_loss)-1, test_acc, 'ro', label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        # Plot training and test loss
        plt.subplot(1, 2, 2)
        plt.plot(train_loss, label='Train Loss')
        plt.plot(len(train_loss)-1, test_loss, 'ro', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()


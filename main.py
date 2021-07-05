"""By Matan Achiel 205642119, Netanel Moyal 307888974"""

import numpy as np
import matplotlib.pyplot as plt
from data import load_dataset
from model import FC_NN
import json
import os


def make_json(W1, W2, b1, b2, id1, id2, activation1, activation2, nn_h_dim, path_to_save):
    """
    make json file with trained parameters.
    W1: numpy arrays of shape (1024, nn_h_dim)
    W2: numpy arrays of shape (nn_h_dim, 1)
    b1: numpy arrays of shape (1, nn_h_dim)
    b2: numpy arrays of shape (1, 1)
    nn_hdim - number of neirons in hidden layer: int
    id1: id1 - str '0123456789'
    id2: id2 - str '0123456789'
    activation1: one of only: 'sigmoid', 'tanh', 'ReLU'
    activation2: one of only: 'sigmoid', 'tanh', 'ReLU'
    """
    trained_dict = {'weights': (W1.tolist(), W2.tolist()),
                    'biases': (b1.tolist(), b2.tolist()),
                    'nn_hdim': nn_h_dim,
                    'activation_1': activation1,
                    'activation_2': activation2,
                    'IDs': (id1, id2)}
    file_path = os.path.join(path_to_save, 'trained_dict_{}_{}'.format(
        trained_dict.get('IDs')[0], trained_dict.get('IDs')[1])
                             )
    with open(file_path, 'w') as f:
        json.dump(trained_dict, f, indent=4)


def plot_results(epoch_avg_loss, acc, epochs, run_info):
    fig, ax = plt.subplots()
    ax.set_title(f"Loss and Accuracy\nFinal Results: Acc = {round(acc[-1], 2)}%, Loss = {round(epoch_avg_loss[-1], 2)}",
                 fontweight="bold")
    # make a plot
    ax.plot(epochs, acc, color="red")
    # set x-axis label
    ax.set_xlabel("Epochs", fontsize=14)
    # set y-axis label
    ax.set_ylabel("Accuracy", color="red", fontsize=14)
    # twin object for two different y-axis on the sample plot

    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(epochs, epoch_avg_loss, color="blue")
    ax2.set_ylabel("Avg loss", color="blue", fontsize=14)
    plt.show()
    # save the plot as a file
    fig.savefig(
        f'results/lr_{run_info["lr"]}_batch_size_{run_info["batch_size"]}_epochs_{len(epochs)}_hidden_{run_info["hidden"]}.jpg',
        format='jpeg',
        dpi=100,
        bbox_inches='tight')


if __name__ == '__main__':
    "Hyper Parameters"
    EPOCH_ACC_CHECK = 35  # var for checking if acc hasnt improve
    EPOCHS = 4
    LR = .4
    BATCH_SIZE = 32
    ACTIVATIONS = ['sigmoid', 'sigmoid']
    LAYER_1_INPUT = 1024
    LAYER_2_INPUT = LAYER_1_INPUT * 3
    LAYER_3_INPUT = 1
    AUGMENTION = False

    """Loading Data"""
    train_data, test_data = load_dataset(augmention=AUGMENTION)

    run_info = {"lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "hidden": LAYER_2_INPUT}

    """Training"""
    nn = FC_NN(layers=[LAYER_1_INPUT, LAYER_2_INPUT, LAYER_3_INPUT], activations=ACTIVATIONS, step_lr=0,
               min_delta_loss=1e-4)
    epoch_avg_loss, acc, final_epochs = nn.train(train_data[0], train_data[1], test_data, epochs=EPOCHS,
                                                 batch_size=BATCH_SIZE, lr=LR, epoch_checker=EPOCH_ACC_CHECK)

    """Accuracy on Validation Data:"""
    print(f'Final Accuracy (On Test) = {round(acc[-1], 2)}%')

    """Plotting:"""
    plot_results(epoch_avg_loss, acc, [i for i in range(final_epochs)], run_info)

    """Creating Required JSON"""
    make_json(nn.weights[0], nn.weights[1], nn.biases[0], nn.biases[1],
              205642119, 307888974, ACTIVATIONS[0], ACTIVATIONS[1],
              LAYER_2_INPUT, os.path.dirname(os.path.realpath(__file__)))

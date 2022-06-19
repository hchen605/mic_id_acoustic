import matplotlib.pyplot as plt

import os
import argparse

import pandas as pd
import re
import numpy as np

def parse_train_log(logfile):
    seed = int(logfile.split("seed")[-1].rstrip(".log"))
    
    with open(logfile, 'r') as f: lines = f.readlines()

    matrics = re.compile(r"loss: \d+.\d+ - accuracy: \d+.\d+ - val_loss: \d+.\d+ - val_accuracy: \d+.\d+")
    num = re.compile(r"\d+\.\d+")
    history = [matrics.search(i) for i in lines]
    history = [num.findall(i.group(0)) for i in history if i is not None]
    history = np.array(history, dtype=np.float64)

    return {"seed": seed, "history": np.array(history, dtype=np.float64)}

def parse_test_log(logfile):
    seed = int(logfile.split("seed")[-1].rstrip(".log"))
    
    with open(logfile, 'r') as f: text= f.read()

    loss = float(re.search(r"--- Test loss: \d+.\d+", text).group(0).split(" ")[-1])
    accuracy = float(re.search(r"- Test accuracy: \d+.\d+", text).group(0).split(" ")[-1])

    return {"seed": seed, "loss": loss, "accuracy": accuracy}

def plot_history(train_log, test_log, step=10):
    history = np.array([i['history'] for i in train_log], dtype=np.float64)
    loss = history[..., 0]
    acc = history[..., 1]
    val_loss = history[..., 2]
    val_acc = history[..., 3]

    fig = plt.figure()
    for j, matrix in enumerate(["loss", "acc", "val_loss", "val_acc"]):
        ax = plt.subplot(2,2,j+1)
        ax.boxplot(history[..., j])
        ax.set_xticks(range(step, loss.shape[1]+1, step))
        ax.set_xticklabels([str(i) for i in range(1, loss.shape[1]+1) if i % step == 0])
        ax.set_title(matrix)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(matrix)
    loss = np.array([i['loss'] for i in test_log])
    acc = np.array([i['accuracy'] for i in test_log])
    fig.suptitle(r"loss=$%.4f\pm%.4f$, acc=$%.4f\pm%.4f$" \
             % (loss.mean(), loss.std(), acc.mean(), acc.std()))
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir-in', type=str, default='log')
    parser.add_argument('--figure-dir-out', type=str, default='figure')
    args = parser.parse_args()

    get_filename_prefix = lambda prefix: [os.path.join(args.log_dir_in, i) \
        for i in os.listdir(args.log_dir_in) if i.startswith(prefix)]
    train_log_3 = [parse_train_log(i) for i in get_filename_prefix('train.0')]
    train_log_18 = [parse_train_log(i) for i in get_filename_prefix('train.1')]
    test_log_3 = [parse_test_log(i) for i in get_filename_prefix('test.0')]
    test_log_18 = [parse_test_log(i) for i in get_filename_prefix('test.1')]

    os.makedirs(args.figure_dir_out, exist_ok=True)
    fig = plot_history(train_log_3, test_log_3)
    fig.suptitle('History of 3 classes classification\n' + fig._suptitle.get_text())
    fig.tight_layout()
    fig.set_size_inches((8,4))
    fig.savefig(os.path.join(args.figure_dir_out, "history_3class.png"))
    
    fig = plot_history(train_log_18, test_log_18)
    fig.suptitle('History of 18 classes classification\n' + fig._suptitle.get_text())
    fig.tight_layout()
    fig.set_size_inches((8,4))
    fig.savefig(os.path.join(args.figure_dir_out, "history_18class.png"))

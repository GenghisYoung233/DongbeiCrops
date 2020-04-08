import sys

sys.path.append("./models")
sys.path.append("..")

import argparse

from argparse import Namespace
import numpy as np

from train import train  # import the train() function from the train.py script


def dict2str(hyperparameter_dict):
    return ",".join([f"{k}={v}" for k,v in hyperparameter_dict.items()])


def tunePARA(modelpara):
    # default parameters
    args = Namespace(
        mode="validation",
        model=modelpara.model,
        epochs=1,
        datapath="../data",
        batchsize=256,
        workers=0,
        device="cpu",
        logdir="/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs" # /home/ga63cuh/Documents/Logs
    )

    while True:
        if args.model == "LSTM":
            args.learning_rate = np.random.uniform(1e-2, 1e-4)
            args.weight_decay = np.random.uniform(1e-2, 1e-8)

            hyperparameter_dict = dict(
                # kernel_size = np.random.choice([3,5,7]),
                # num_classes=,
                hidden_dims=np.random.choice([32, 64, 128]),
                # num_layers=np.random.choice([1, 2, 3, 4]),
                dropout=np.random.uniform(0, 0.8)
            )

        elif args.model == "MSResNet":
            args.learning_rate = np.random.uniform(1e-2, 1e-4)
            args.weight_decay = np.random.uniform(1e-2, 1e-8)

            hyperparameter_dict = dict(
                # input_dim=,
                # num_classes=,
                # layers=,
                hidden_dims=np.random.choice([32, 64, 128])
            )

        elif args.model == "TransformerEncoder":
            args.learning_rate = np.random.uniform(1e-2, 1e-4)
            args.weight_decay = np.random.uniform(1e-2, 1e-8)

            hyperparameter_dict = dict(
                # n_head=8,
                # n_layers=6,
                dropout=np.random.uniform(0, 0.8)
                # input_dim=10,
                # len_max_seq=100,
                # d_word_vec=512,
                # d_model=512,
                # d_inner=2048,
                # d_k=64,
                # d_v=64,
                # num_classes=6
            )

        elif args.model == "TempCNN":
            args.learning_rate = np.random.uniform(1e-2, 1e-4)
            args.weight_decay = np.random.uniform(1e-2, 1e-8)

            hyperparameter_dict = dict(
                kernel_size=np.random.choice([3, 5, 7]),
                hidden_dims=np.random.choice([32, 64, 128]),
                dropout=np.random.uniform(0, 0.8)
                # num_layers = np.random.choice([1,2,3,4])
            )

        else:
            raise ValueError("invalid model argument. choose from 'LSTM','MSResNet','TransformerEncoder', or 'TempCNN'")

        args.hyperparameter = hyperparameter_dict
        hyperparameter_string = dict2str(hyperparameter_dict)

        # define a descriptive model name that contains all the hyperparameters - change to /home/ga63cuh/Documents/Logs - trainlog namechange?
        if args.model == "LSTM":
            args.store = f"/tmp/LSTM-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/LSTM-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "MSResNet":
            args.store = f"/tmp/MSResNet-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/MSResNet-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "TransformerEncoder":
            args.store = f"/tmp/TransformerEncoder-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/TransformerEncoder-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "TempCNN":
            args.store = f"/tmp/TempCNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/TempCNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        train(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='select model architecture')
    args = parser.parse_args()
    return args


modelpara = parse_args()
tunePARA(modelpara)
import os
import sys
import argparse
import numpy as np
import pandas as pd
import math
import random
import itertools
import csv
import time
import rasterio as rio
import rasterio.windows
import sklearn.metrics
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from models import *


def main(model_name, train_file, test_file, result_path, device, epoch, feature, n_channels, patch_size, batch_size_list, val_rate, monitor, hyperparameter, learning_rate, weight_decay, input_raster, result_file, proba, transfer_learning):
    # -- Creating output path if does not exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # ---- output files
    result_path = result_path + '/' + model_name + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print("Model: ", model_name)

    # Load datasets
    X_train, polygon_ids_train, y_train = readSITSData(train_file)
    X_test, polygon_ids_test, y_test = readSITSData(test_file)

    n_classes_test = len(np.unique(y_test))
    n_classes_train = len(np.unique(y_train))
    if (n_classes_test != n_classes_train):
        print("WARNING: different number of classes in train and test")
    n_classes = max(n_classes_train, n_classes_test)

    # Torch, numpy, whatever, all index from 0, if we did not assign landcover classes
    # with [0, 1, 2, 3, ...], it may cause problem, things get easier by reindex classes
    lc_ids_old = np.unique(y_train)
    lc_ids_old.sort()
    lc_ids_new = np.arange(n_classes_train)

    indexes = [np.where(y_train == lc_id)[0] for lc_id in lc_ids_old]
    for index, lc_id_new in zip(indexes, lc_ids_new):
        y_train[index] = lc_id_new

    indexes = [np.where(y_test == lc_id)[0] for lc_id in lc_ids_old]
    for index, lc_id_new in zip(indexes, lc_ids_new):
        y_test[index] = lc_id_new

    relation = np.vstack((lc_ids_old, lc_ids_new))

    y_train_one_hot = np.eye(n_classes, dtype='uint8')[y_train]
    y_test_one_hot = np.eye(n_classes, dtype='uint8')[y_test]

    X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1] / n_channels), n_channels)
    X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1] / n_channels), n_channels)

    # ---- Normalizing the data per band,
    min_per = np.percentile(X_train, 2, axis=(0, 1))
    max_per = np.percentile(X_train, 100-2, axis=(0, 1))
    X_train = (X_train-min_per)/(max_per-min_per)
    X_test = (X_test-min_per)/(max_per-min_per)

    # ---- Save min_max
    minMaxVal_file = result_path + 'min_Max.txt'
    save_minMaxVal(minMaxVal_file, min_per, max_per)

    # ---- Extracting a validation set (if necesary)
    if val_rate > 0:
        X_train, y_train, X_val, y_val = extractValSet(X_train, polygon_ids_train, y_train, val_rate)
        # --- Computing the one-hot encoding (recomputing it for train)
        y_train_one_hot = np.eye(n_classes, dtype='uint8')[y_train]
        y_val_one_hot = np.eye(n_classes, dtype='uint8')[y_val]

    input_dim = X_train.shape[2]
    num_classes = n_classes_train
    sequencelength = X_train.shape[1]
    device = torch.device(device)
    model_save_path = result_path + '/' + 'Best_model'
    minmax = [min_per, max_per]

    # Specify epoch to train the model
    if epoch:
        # Get model returned by choosed function and train it
        model = get_model(model_name, input_dim, num_classes, sequencelength, device, **hyperparameter)
        train_epoch(model, X_train, y_train, X_test, y_test, model_save_path, device, batch_size_list, epoch, monitor, learning_rate, weight_decay)

    # Load model and classify raster
    if input_raster and result_file:
        model_path = model_save_path
        model = get_model(model_name, input_dim, num_classes, sequencelength, device, **hyperparameter)
        model.load_state_dict(torch.load(model_path))
        classify_image(model, input_raster, result_file, device, batch_size_list, n_channels, patch_size, minmax=minmax, proba=proba, n_classes=n_classes, relation=relation)

    # Evaluate the performence of transfer learning
    if transfer_learning:
        model_path = model_save_path
        model = get_model(model_name, input_dim, num_classes, sequencelength, device, **hyperparameter)
        model.load_state_dict(torch.load(model_path))
        transfer_evaluate(model, X_test, X_test, result_path, device, batch_size_list)

def train_epoch(model, X_train, y_train, X_test, y_test, model_save_path, device, batch_size_list=[4096,8192,8192], epoch=20, monitor="test_loss", learning_rate=0.001, weight_decay=0):
    # covert numpy to pytorch tensor and put into gpu
    X_train = torch.from_numpy(X_train.astype(np.float32))
    if sys.platform == "win32":
        y_train = torch.from_numpy(y_train.astype(np.int64))
    elif "linux" in sys.platform:
        y_train = torch.from_numpy(y_train.astype(np.int_))

    # add channel dimension to time series data
    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze_(1)

    # ---- optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, min_lr=0.0001)

    # build dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size_list[0], shuffle=True)

    early_stopping = EarlyStopping(patience=0, path=model_save_path)

    log = list()
    start = time.time()
    for epoch in range(epoch):
        model.train()
        for sample in tqdm(train_loader, desc=f'epoch {epoch}'):
            optimizer.zero_grad()
            log_proba = model.forward(sample[0].to(device))
            output = criterion(log_proba, sample[1].to(device))
            output.backward()
            optimizer.step()
        scheduler.step(output)
        train_loss = output.item()

        # get test loss
        model.eval()
        test_loss, y_true, y_pred, y_score = evaluate(model, X_test, y_test, device, batch_size_list)

        Classes = [f'class {i}' for i in np.unique(y_true.cpu())]
        scores = metrics(y_true.cpu(), y_pred.cpu(), Classes)
        scores_msg = ", ".join([f"{k}={v}" for (k, v) in scores.items()])
        test_loss = test_loss.cpu().detach().numpy()[0]

        scores["epoch"] = epoch
        scores["train_loss"] = train_loss
        scores["test_loss"] = test_loss
        scores["time"] = (time.time() - start) / 60
        log.append(scores)
        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(os.path.join(os.path.dirname(model_save_path), "trainlog.csv"))

        print(f'train_loss={train_loss}, test_loss={test_loss}, kappa={scores["kappa"]}\n', scores["report"])  # In report, precision means User_accuracy, recall means Producer_accuracy

        # if kapp < 0.01, there is no need to train any more
        if scores["kappa"] < 0.01 and epoch >= 1:
            print("training terminated for no accuray")
            break

        # early_stopping needs the monitor to check if it has improved,
        # and if it has, it will make a checkpoint of the current model
        score = scores[monitor]
        if "loss" in monitor:
            score = -score

        early_stopping(score, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(log[-2]["confusion_matrix"])  # Be aware, scikit-learn put (true_label, pred_label) in (Y, X) shape which is reversed from ArcGIS and ENVI
    # Uncomment to save model of last epoch, comment to save mode before early stopping
    # torch.save(model.state_dict(), model_save_path)

def evaluate(model, X, y, device, batch_size_list=[4096,8192,8192]):
    # covert numpy to pytorch tensor and put into gpu
    X = torch.from_numpy(X.astype(np.float32))
    if sys.platform == "win32":
        y = torch.from_numpy(y.astype(np.int64))
    elif "linux" in sys.platform:
        y = torch.from_numpy(y.astype(np.int_))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size_list[1], shuffle=False)

    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()
        for x, y_true in tqdm(dataloader):
            log_proba = model.forward(x.to(device))
            loss = criterion(log_proba, y_true.to(device))
            losses.append(loss)
            y_true_list.append(y_true)
            y_pred_list.append(log_proba.argmax(-1))
            y_score_list.append(log_proba.exp())

    return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list)


def inference(model, X, n_classes, device, batch_size_list=[4096,8192,8192]):
    X = torch.from_numpy(X.astype(np.float32))
    dataloader = DataLoader(X, batch_size_list[2], shuffle=False)

    model.eval()
    with torch.no_grad():
        y_predict_array = np.zeros(shape=(X.shape[0], n_classes), dtype=np.float32)
        for i, batch in enumerate(dataloader):
            log_proba = model.forward(batch.to(device))
            y_predict_array[i*batch_size_list[2]:(i+1)*batch_size_list[2], :] = log_proba.cpu()
    return y_predict_array


def transfer_evaluate(model, X, y, log_path, device, batch_size_list=[4096,8192,8192]):
    model.eval()
    test_loss, y_true, y_pred, y_score = evaluate(model, X, y, device, batch_size_list)

    Classes = [f'class {i}' for i in np.unique(y_true.cpu())]
    scores = metrics(y_true.cpu(), y_pred.cpu(), Classes)
    scores_msg = ", ".join([f"{k}={v}" for (k, v) in scores.items()])
    test_loss = test_loss.cpu().detach().numpy()[0]

    scores["test_loss"] = test_loss
    log.append(scores)
    log_df = pd.DataFrame(log)
    log_df.to_csv(os.path.join(log_path, "transfer_log.csv"))
    print(scores["confusion_matrix"])
    print(scores["report"])


def metrics(y_true, y_pred, Classes):
    report = sklearn.metrics.classification_report(y_true, y_pred, target_names=Classes, digits=6, zero_division=1)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro", zero_division=1)
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted", zero_division=1)

    return dict(
        report = report,
        confusion_matrix=confusion_matrix,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
    )

def get_model(model, ndims, num_classes, sequencelength, device, **hyperparameter):
    if model == "OmniScaleCNN":
        model = OmniScaleCNN(input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength, **hyperparameter).to(device)
    elif model == "LSTM":
        model = LSTM(input_dim=ndims, num_classes=num_classes, **hyperparameter).to(device)
    elif model == "StarRNN":
        model = StarRNN(input_dim=ndims,
                        num_classes=num_classes,
                        bidirectional=False,
                        use_batchnorm=False,
                        use_layernorm=True,
                        device=device,
                        **hyperparameter).to(device)
    elif model == "InceptionTime":
        model = InceptionTime(input_dim=ndims, num_classes=num_classes, device=device,
                              **hyperparameter).to(device)
    elif model == "MSResNet":
        model = MSResNet(input_dim=ndims, num_classes=num_classes, **hyperparameter).to(device)
    elif model == "TransformerModel":
        model = TransformerModel(input_dim=ndims, num_classes=num_classes, activation="relu", **hyperparameter).to(device)
    elif model == "TempCNN":
        model = TempCNN(input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength, **hyperparameter).to(device)
    else:
        raise ValueError("invalid model argument.")

    return model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, score, model):

        # run "if" for the first batch, run "elif" or "else" after
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Saves model'''
        torch.save(model.state_dict(), self.path)

def readSITSData(name_file):
    """
        Read the data contained in name_file
        INPUT:
            - name_file: file where to read the data
        OUTPUT:
            - X: variable vectors for each example
            - polygon_ids: id polygon (use e.g. for validation set)
            - Y: label for each example
    """

    data = pd.read_csv(name_file, sep=',', header=None)

    y_data = data.iloc[:, 0]
    y = np.asarray(y_data.values, dtype='uint8')

    polygonID_data = data.iloc[:, 1]
    polygon_ids = polygonID_data.values
    polygon_ids = np.asarray(polygon_ids, dtype='uint64')

    X_data = data.iloc[:, 2:]
    X = np.asarray(X_data.values, dtype='int32')

    return X, polygon_ids, y


def addingfeat_reshape_data(X, feature_strategy, nchannels):
    """
        Reshaping (feature format (3 bands): d1.b1 d1.b2 d1.b3 d2.b1 d2.b2 d2.b3 ...)
        INPUT:
            -X: original feature vector ()
            -feature_strategy: used features (options: SB, NDVI, SB3feat)
            -nchannels: number of channels
        OUTPUT:
            -new_X: data in the good format for Keras models
    """

    if feature_strategy == 'SB':
        print("SPECTRAL BANDS-----------------------------------------")
        return X.reshape(X.shape[0], int(X.shape[1] / nchannels), nchannels)

    elif feature_strategy == 'NDVI':
        print("NDVI only----------------------------------------------")
        new_X = computeNDVI(X, nchannels)
        return np.expand_dims(new_X, axis=2)

    elif feature_strategy == 'SB3feat':
        print("SB + NDVI + NDWI + IB----------------------------------")
        NDVI, NDWI, IB = addFeatures(X)
        new_X = X.reshape(X.shape[0], int(X.shape[1] / nchannels), nchannels)
        new_X = np.dstack((new_X, NDVI))
        new_X = np.dstack((new_X, NDWI))
        new_X = np.dstack((new_X, IB))
        return new_X
    else:
        print("Not referenced!!!-------------------------------------------")
        return -1


def read_minMaxVal(minmax_file):
    with open(minmax_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        min_per = next(reader)
        max_per = next(reader)
    min_per = [float(k) for k in min_per]
    min_per = np.array(min_per)
    max_per = [float(k) for k in max_per]
    max_per = np.array(max_per)
    return min_per, max_per


def save_minMaxVal(minmax_file, min_per, max_per):
    with open(minmax_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(min_per)
        writer.writerow(max_per)


def extractValSet(X_train, polygon_ids_train, y_train, val_rate=0.2):
    unique_pol_ids_train, indices = np.unique(polygon_ids_train, return_inverse=True)  # -- pold_ids_train = unique_pol_ids_train[indices]
    nb_pols = len(unique_pol_ids_train)
    ind_shuffle = list(range(nb_pols))
    random.shuffle(ind_shuffle)
    list_indices = [[] for i in range(nb_pols)]
    shuffle_indices = [[] for i in range(nb_pols)]
    [list_indices[ind_shuffle[val]].append(idx) for idx, val in enumerate(indices)]

    final_ind = list(itertools.chain.from_iterable(list_indices))
    m = len(final_ind)
    final_train = int(math.ceil(m * (1.0 - val_rate)))
    shuffle_polygon_ids_train = polygon_ids_train[final_ind]
    id_final_train = shuffle_polygon_ids_train[final_train]

    while shuffle_polygon_ids_train[final_train - 1] == id_final_train:
        final_train = final_train - 1

    new_X_train = X_train[final_ind[:final_train], :, :]
    new_y_train = y_train[final_ind[:final_train]]
    new_X_val = X_train[final_ind[final_train:], :, :]
    new_y_val = y_train[final_ind[final_train:]]

    return new_X_train, new_y_train, new_X_val, new_y_val


# This code was implemented from http://devseed.com/sat-ml-training/Randomforest_cropmapping-with_GEE#Model-training
def classify_image(model, input_raster, result_file, device, batch_size_list, n_channels, patch_size=500, minmax=[], proba=False, feature="SB", n_classes=None, relation=None):
    # in this case, we predict over the entire input image
    src = rio.open(input_raster, 'r')
    profile = src.profile
    nodata = profile["nodata"]
    profile.update(
        dtype=rio.uint8,
        count=1,
        nodata=0,
    )
    profile_conf = src.profile
    profile_conf.update(
        dtype=rio.float32,
        count=n_classes,
        nodata=9999,
    )


    result_conf_file = result_file.replace('.tif', '_confmap.tif')
    dst = rio.open(result_file, 'w', **profile)
    dst_conf = rio.open(result_conf_file, 'w', **profile_conf)
    for i in tqdm(range((src.shape[0] // patch_size) + 1)):
        for j in range((src.shape[1] // patch_size) + 1):
            # define the pixels to read (and write) with rasterio windows reading
            window = rasterio.windows.Window(
                j * patch_size,
                i * patch_size,
                # don't read past the image bounds
                min(patch_size, src.shape[1] - j * patch_size),
                min(patch_size, src.shape[0] - i * patch_size))

            # read the image into the proper format
            X = src.read(window=window)
            X = np.moveaxis(X, 0, 2)
            height, width, bands = X.shape

            X = X.reshape(height * width, int(bands / n_channels), n_channels)
            # Locate the backgraound pixels
            nodata_rows = np.where((X == nodata))[0]
            # ---- pre-processing the data
            min_per, max_per = minmax
            X = (X - min_per) / (max_per - min_per)

            # skip empty inputs
            if not len(X):
                continue
            # predict
            log_proba = inference(model, X, n_classes, device, batch_size_list)
            pred_y = log_proba.argmax(axis=-1)

            indexes = [np.where(pred_y == lc_id_new)[0] for lc_id_new in relation[1, :]]
            for index, lc_id_old in zip(indexes, relation[0, :]):
                pred_y[index] = lc_id_old
            # Nodata always nodata
            pred_y[nodata_rows] = 0
            # Reshape to original size
            pred_y = pred_y.reshape(height, width)
            # write to the final files
            dst.write(pred_y.astype(rio.uint8), 1, window=window)

            if proba:
                log_proba[nodata_rows, :] = 9
                confpred_y = np.exp(log_proba)
                confpred_y[np.where(confpred_y > 1.0)[0]] = 9999
                confpred_y = confpred_y.reshape(height, width, n_classes)
                for b in range(n_classes):
                    dst_conf.write(confpred_y[:, :, b].astype(rio.float32), b+1, window=window)

    # Write to disk
    dst.close()
    dst_conf.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Running deep learning architectures on SITS datasets')
    parser.add_argument('--model', type=str,
                        help='choose model', default="TransformerModel")
    parser.add_argument('--train_file', dest='train_file',
                        help='path for train datasets',
                        default=None)
    parser.add_argument('--test_file', dest='test_file',
                        help='path for test datasets',
                        default=None)
    parser.add_argument('--result_path', dest='result_path',
                        help='path where to store the trained model',
                        default=None)
    parser.add_argument('--device', dest='device',
                        help='torch.Device. either "cpu" or "cuda", default is "cpu"',
                        default="cpu")
    parser.add_argument('--epoch', dest='epoch', type=int,
                        help='train epoch', default=None)
    parser.add_argument('--feat', dest='feature',
                        help='used feature vector',
                        default="SB")
    parser.add_argument('--n_channels', dest='n_channels', type=int,
                        help='channel number', default=8)
    parser.add_argument('--patch_size', dest='patch_size', type=int,
                        help="define the pixels to read (and write) with rasterio windows reading",
                        default=500)
    parser.add_argument('--batch_size_list', dest='batch_size_list',
                        type=lambda x: [int(b) for b in x.split(',')],
                        help='list of batch size for train, evaluate, inference',
                        default="128,128,8292")
    parser.add_argument('--val_rate', dest='val_rate', type=float,
                        help='validation rate', default=0)
    parser.add_argument('--monitor', dest='monitor', type=str,
                        help='Stop training when this metric stop improving. Options: test_loss, train_loss, kappa, accuracy, recall_micro, precision_micro, f1_micro',
                        default="kappa")
    parser.add_argument('--hyperparameter', dest='hyperparameter', type=str,
                        help='model specific hyperparameter as single string, separated by comma of format param1=value1,param2=value2',
                        default=None)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        help='Learning rate for model training',
                        default=1e-3)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        help='Weight decay for model training',
                        default=1e-6)
    parser.add_argument('--input_raster', dest='input_raster',
                        help='path for input raster',
                        default=None)
    parser.add_argument('--result_file', dest='result_file',
                        help='path for classification result',
                        default=None)
    parser.add_argument('--proba', dest='proba',
                        help='if True, probabilities are stored',
                        default=False, action="store_true")
    parser.add_argument('--transfer_learning', dest='transfer_learning',
                        help='if True, start transfer learning mode',
                        default=False, action="store_true")

    args = parser.parse_args()
    hyperparameter_dict = dict()
    if args.hyperparameter and args.hyperparameter != "None":
        for hyperparameter_string in args.hyperparameter.split(","):
            param, value = hyperparameter_string.split("=")
            if value in ["True", "False"]:
                hyperparameter_dict[param] = True
            elif "." in value:
                hyperparameter_dict[param] = float(value)
            else:
                hyperparameter_dict[param] = int(value)
    args.hyperparameter = hyperparameter_dict

    main(args.model, args.train_file, args.test_file, args.result_path, args.device, args.epoch, args.feature, args.n_channels, args.patch_size, args.batch_size_list, args.val_rate, args.monitor, args.hyperparameter, args.learning_rate, args.weight_decay, args.input_raster, args.result_file, args.proba, args.transfer_learning)


import os
import sys
import argparse
import math
import random
import time
import csv
import gdal, osr, ogr
from gdalconst import *
import psutil
import fiona
import PySimpleGUI as sg
import rasterio as rio
import rasterio.windows
import rasterio.mask
import numpy as np
import pandas as pd
import joblib
import geopandas as gpd
from tqdm import tqdm
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def main(model, input_training_raster, train_feature, input_test_raster, test_feature, input_test_csv, result_path, n_channels, n_jobs, model_path, raster_to_classify, patch_size, output_raster, train_ratio, n_estimators, max_depth, max_num_of_samples_per_class):
    # -- Creating output path if does not exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # ---- output files
    result_path = os.path.join(result_path, model)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print("Model: ", model)
    # Generatin train/test datasets
    train_list, test_list, _ = split_train_feature(train_feature, train_ratio)
    train_data = generate_training_data(input_training_raster, train_feature, train_list, max_num_of_samples_per_class)
    X_train, y_train = train_data[:, 1:], train_data[:, 0]

    if input_test_raster and test_feature:
        _, test_list, _ = split_train_feature(test_feature, train_ratio=0)
        test_data = generate_training_data(input_test_raster, test_feature, test_list)
        X_test, y_test = test_data[:, 1:], test_data[:, 0]
    elif input_test_csv:
        df = pd.read_csv(input_test_csv, sep=',', header=None)
        test_data = np.asarray(df.values)
        X_test, y_test = test_data[:, 2:], test_data[:, 0]
    else:
        test_data = generate_training_data(input_training_raster, train_feature, test_list, max_num_of_samples_per_class)
        X_test, y_test = test_data[:, 1:], test_data[:, 0]

    # Fitting the classifier into the Training set
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

    if model in ["RF", "SVM"]:
        is_ts = False
        # ---- Normalizing the data per band,
        min_per = np.percentile(X_train, 2, axis=(0))
        max_per = np.percentile(X_train, 100 - 2, axis=(0))
        X_train = (X_train - min_per) / (max_per - min_per)
        X_test = (X_test - min_per) / (max_per - min_per)

        if model == "RF":
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion='entropy', random_state=None, verbose=0, n_jobs=n_jobs)

        elif model == "SVM":
            clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', cache_size=200), max_samples=1.0, n_estimators=n_estimators, verbose=0, n_jobs=n_jobs))

    elif model == "RF_TS":
        from sktime.classification.compose import TimeSeriesForestClassifier
        from sktime.transformations.panel.compose import ColumnConcatenator

        is_ts = True

        X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1] / n_channels), n_channels)
        X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1] / n_channels), n_channels)

        # ---- Normalizing the data per band,
        min_per = np.percentile(X_train, 2, axis=(0, 1))
        max_per = np.percentile(X_train, 100 - 2, axis=(0, 1))
        X_train = (X_train - min_per) / (max_per - min_per)
        X_test = (X_test - min_per) / (max_per - min_per)

        steps = [
            ("concatenate", ColumnConcatenator()),
            ("classify", TimeSeriesForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs)),
        ]
        clf = Pipeline(steps)

    # Train classifier
    clf.fit(X_train, y_train)
    # Save trained classifier
    if not model_path:
        model_path = os.path.join(result_path, 'Best_model.pkl')
    joblib.dump(clf, model_path)

    # Evaluation
    start = time.time()
    y_pred = clf.predict(X_test)

    Classes = [f'class {i}' for i in np.unique(y_test)]
    scores = metrics(y_test, y_pred, Classes)
    scores_msg = ", ".join([f"{k}={v}" for (k, v) in scores.items()])

    scores["time"] = (time.time() - start) / 60

    log = {k: [v] for k, v in scores.items()}
    log_df = pd.DataFrame(log)
    log_df.to_csv(os.path.join(result_path, "trainlog.csv"))

    print(scores["report"])  # In report, precision means User_accuracy, recall means Producer_accuracy
    print(scores["confusion_matrix"])

    # ---- Save min_max
    minMaxVal_file = os.path.join(result_path, 'min_Max.txt')
    save_minMaxVal(minMaxVal_file, min_per, max_per)

    # Inference on raster
    if raster_to_classify:
        classify_image(raster_to_classify, model_path, output_raster, n_channels, patch_size=patch_size, minmax=[min_per, max_per], is_ts=is_ts, relation=relation)


def split_train_feature(train_feature, train_ratio=0.8, predict_ratio=0):
    # read train_feature, get classvalue
    shp = fiona.open(train_feature, "r")
    classvalues = [int(feature['properties']['Classvalue']) for feature in shp]

    train_list = []
    test_list = []
    predict_list = []

    for classvalue in list(set(classvalues)):
        fid_list = []
        for i in range(len(classvalues)):
            if classvalues[i] == classvalue:
                fid_list.append(i)

        random.shuffle(fid_list)
        train = fid_list[:int(len(fid_list) * train_ratio)]
        test = fid_list[int(len(fid_list) * train_ratio):int(len(fid_list) * (1 - predict_ratio))]
        predict = fid_list[int(len(fid_list) * (1 - predict_ratio)):int(len(fid_list) * 1)]
        [train_list.append(i) for i in train]
        [test_list.append(i) for i in list(test)]
        [predict_list.append(i) for i in list(predict)]
    return train_list, test_list, predict_list


def generate_training_data(input_training_raster, train_feature, sample_list,  max_num_of_samples_per_class=1000):
    # read input_training_raster and train_feature
    src = rio.open(input_training_raster)
    shp = fiona.open(train_feature, "r")
    # Reproject train_feature if the crs are different
    if src.crs != shp.crs:
        shp = gpd.read_file(train_feature)
        shp = shp.to_crs(f"{src.crs}")
        train_feature = train_feature.replace(".shp", "_reprojected.shp")
        shp.to_file(train_feature)
        shp = fiona.open(train_feature, "r")

    # generate empty array to store data, reduce array size if out of memory
    shape = (int(src.width * src.height), src.count + 1)
    while True:
        try:
            out_images = np.zeros(shape=shape, dtype=np.float32)
        except:
            shape = (int(shape[0] / 2), shape[1])
        else:
            break

    row0 = 0
    # loop through every feature(polygon) to extract pixels' value within it.
    for i in tqdm(sample_list, desc="Generating train/test datasets"):
        # Masking raster using feature
        shape = [shp[i]["geometry"]]
        out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
        # Reshape array to (n_pixels, n_bands)
        out_image = out_image.reshape(-1).reshape(-1, src.count, order='F')
        # Delete nodata rows
        nodata_row = np.unique(np.where(out_image == -9999)[0])
        out_image = np.delete(out_image, nodata_row, axis=0)
        # Add classvalue to every row
        classvalue = int(shp[i]['properties']['Classvalue'])
        lc_id = np.full((out_image.shape[0]), classvalue).reshape(-1, 1)
        out_image = np.append(lc_id, out_image, axis=1)
        # Write in out_image
        row1 = row0 + out_image.shape[0]
        out_images[row0:row1, :] = out_image
        row0 += out_image.shape[0]

    # Delete zero_rows
    zero_rows = np.where((out_images[:, 1:] == 0).all(axis=1))[0]
    out_images = np.delete(out_images, zero_rows, axis=0)
    # Get unique landcover classvalue
    lc_classvalues = np.unique(out_images[:, 0])
    # Randomly choose rows in each landcover class
    datasets = np.empty([0, out_images.shape[1]], dtype=np.int64)
    for value in lc_classvalues:
        out_image = np.take(out_images, np.where(out_images[:, 0] == value)[0], axis=0)
        indices = np.floor(np.arange(1, out_image.shape[0], out_image.shape[0] / max_num_of_samples_per_class)).astype(np.int64)
        out_image = np.take(out_image, indices, axis=0)
        datasets = np.append(datasets, out_image, axis=0)

    np.random.shuffle(datasets)
    return datasets

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

def save_minMaxVal(minmax_file, min_per, max_per):
	with open(minmax_file, 'w') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(min_per)
		writer.writerow(max_per)

# This function was implemented from http://devseed.com/sat-ml-training/Randomforest_cropmapping-with_GEE#Model-training
def classify_image(raster_to_classify, model_path, output_raster, n_channels, patch_size=500, minmax=[], relation=None, is_ts=True):
    # in this case, we predict over the entire input image
    # (only small portions were used for training)
    clf = joblib.load(model_path)

    src = rio.open(raster_to_classify, 'r')
    profile = src.profile
    profile.update(
        dtype=rio.uint8,
        count=1,
    )
    
    dst = rio.open(output_raster, 'w', **profile)
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

            if is_ts:
                X = X.reshape(height * width, int(bands / n_channels), n_channels)
            else:
                X = X.reshape(height * width, X.shape[2])
            # Locate the backgraound pixels
            nodata_rows = np.where((X == profile["nodata"]))[0]
            # ---- pre-processing the data
            min_per, max_per = minmax
            X = (X - min_per) / (max_per - min_per)
            # predict
            pred_y = clf.predict(X)
            indexes = [np.where(pred_y == lc_id_new)[0] for lc_id_new in relation[1, :]]
            for index, lc_id_old in zip(indexes, relation[0, :]):
                pred_y[index] = lc_id_old
            # Nodata always nodata
            pred_y[nodata_rows] = 0
            # Reshape to original size
            pred_y = pred_y.reshape(height, width)

            # write to the final files
            dst.write(pred_y.astype(rio.uint8), 1, window=window)
    # Write to disk
    dst.close()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str,
                        help='Choose between RF(random forest), SVM(suppport vector machine), RF_TS(Random forest for time-series)',
                        default="RF")
    parser.add_argument('--input_training_raster', dest='input_training_raster',
                        help='path for input raster for training',
                        default=None)
    parser.add_argument('--train_feature', dest='train_feature',
                        help='path for train feature',
                        default=None)
    parser.add_argument('--input_test_raster', dest='input_test_raster',
                        default=None)
    parser.add_argument('--test_feature', dest='test_feature',
                        help='path for test feature',
                        default=None)
    parser.add_argument('--input_test_csv', dest='input_test_csv',
                        help='path for test csv',
                        default=None)
    parser.add_argument('--result_path', dest='result_path',
                        help='path where to store the trained model',
                        default=None)
    parser.add_argument('--n_channels', dest='n_channels', type=int,
                        help='channel number', default=8)
    parser.add_argument('--n_jobs', dest='n_jobs', type=int,
                        help='number of thread', default=-1)
    parser.add_argument('--model_path', dest='model_path',
                        default=None)
    parser.add_argument('--raster_to_classify', dest='raster_to_classify',
                        default=None)
    parser.add_argument('--patch_size', dest='patch_size', type=int,
                        help="define the pixels to read (and write) with rasterio windows reading",
                        default=500)
    parser.add_argument('--output_raster', dest='output_raster',
                        default=None)
    parser.add_argument('--train_ratio', dest='train_ratio', type=float,
                        help='ratio of train part', default=0.8)
    parser.add_argument('--n_estimators', dest='n_estimators', type=int,
                        help='number of trees', default=500)
    parser.add_argument('--max_depth', dest='max_depth', type=int,
                        help='max number of trees depth', default=25)
    parser.add_argument('--max_num_of_samples_per_class', dest='max_num_of_samples_per_class', type=int,
                        help='Max number of samples per class ', default=1000)

    args = parser.parse_args()
    return args


def GUI():
    # Random theme for GUI
    theme = random.choice(sg.theme_list())
    sg.theme(theme)  # A touch of color
    SYMBOL_UP = '▲'
    SYMBOL_DOWN = '▼'

    def collapse(layout, key):
        return sg.pin(sg.Column(layout, key=key))

    section_train = [[sg.Text('input training raster', size=(15, 1), font=("Helvetica"), auto_size_text=False, justification='left', tooltip="Path for input raster for training"), sg.InputText(), sg.FileBrowse()],
                     [sg.Text('train feature', size=(15, 1), font=("Helvetica"), auto_size_text=False, justification='left', tooltip="Path for train feature"), sg.InputText(), sg.FileBrowse()],
                     [sg.Text('result path', size=(15, 1), font=("Helvetica"), auto_size_text=False, justification='left', tooltip="Path where to store the trained model"), sg.InputText(), sg.FolderBrowse()],
                     [sg.Text('number of trees', font=("Helvetica"), tooltip="Number of trees"), sg.InputText("200", size=(10, 1))],
                     [sg.Text('max depth', font=("Helvetica"), tooltip="Max number of trees depth"), sg.InputText("25", size=(10, 1))],
                     [sg.Text('max numer of samples per class', font=("Helvetica"), tooltip="Max number of samples per class"), sg.InputText("1000", size=(10, 1))],
                     ]

    section_inference = [[sg.Text('raster to classify', size=(15, 1), font=("Helvetica"), auto_size_text=False, justification='left', tooltip="the raster you want to classify"), sg.InputText(), sg.FileBrowse()],
                         [sg.Text('output raster', size=(15, 1), font=("Helvetica"), auto_size_text=False, justification='left', tooltip="Path to store the classified raster"), sg.InputText(), sg.FileBrowse()],
                         [sg.Text('model path(optional)', size=(16, 1), font=("Helvetica"), auto_size_text=False, justification='left', tooltip="Path for trained model, required when result path not provided"), sg.InputText(), sg.FileBrowse()],
                         ]

    section1 = [[sg.Input('Input sec 1', key='-IN1-')],
                [sg.Input(key='-IN11-')],
                [sg.Button('Button section 1', button_color='yellow on green'),
                 sg.Button('Button2 section 1', button_color='yellow on green'),
                 sg.Button('Button3 section 1', button_color='yellow on green')]]

    section2 = [[sg.I('Input sec 2', k='-IN2-')],
                [sg.I(k='-IN21-')],
                [sg.B('Button section 2', button_color=('yellow', 'purple')),
                 sg.B('Button2 section 2', button_color=('yellow', 'purple')),
                 sg.B('Button3 section 2', button_color=('yellow', 'purple'))]]

    layout = [[sg.T(SYMBOL_DOWN, enable_events=True, k='-OPEN SEC1-', text_color='yellow'), sg.T('Train', enable_events=True, text_color='yellow', k='-OPEN SEC1-TEXT')],
              [collapse(section_train, '-SEC1-')],
              [sg.T(SYMBOL_DOWN, enable_events=True, k='-OPEN SEC2-', text_color='purple'),
               sg.T('Inference', enable_events=True, text_color='purple', k='-OPEN SEC2-TEXT')],
              [collapse(section_inference, '-SEC2-')],
              [sg.Button('OK', font=("Helvetica", 15), tooltip='Click to submit this window'), sg.Cancel(font=("Helvetica", 15))]]

    window = sg.Window('Random forest classifier for remote sensed data', layout)

    opened1, opened2 = True, True

    event = None
    while True:  # Event Loop
        event, values = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Cancel':
            os._exit(1)
        elif event == 'OK':
            break

        if event.startswith('-OPEN SEC1-'):
            opened1 = not opened1
            window['-OPEN SEC1-'].update(SYMBOL_DOWN if opened1 else SYMBOL_UP)
            window['-SEC1-'].update(visible=opened1)

        if event.startswith('-OPEN SEC2-'):
            opened2 = not opened2
            window['-OPEN SEC2-'].update(SYMBOL_DOWN if opened2 else SYMBOL_UP)
            # window['-OPEN SEC2-CHECKBOX'].update(not opened2)
            window['-SEC2-'].update(visible=opened2)

    window.close()
    return [values[i] for i in values]


if __name__ == "__main__":
    args = argument_parser()

    # If there is no input from console, open graphic user interface
    if not args.input_training_raster and not args.train_feature:
        args.input_training_raster, _, args.train_feature, _, args.result_path, _, n_estimators, max_depth, max_num_of_samples_per_class, _, args.raster_to_classify, _, args.output_raster, _, args.model_path, *_ = GUI()
        args.n_estimators = int(n_estimators)
        args.max_depth = int(max_depth)
        args.max_num_of_samples_per_class = int(max_num_of_samples_per_class)

    # Start
    main(args.model, args.input_training_raster, args.train_feature, args.input_test_raster, args.test_feature, args.input_test_csv, args.result_path, args.n_channels, args.n_jobs, args.model_path, args.raster_to_classify, args.patch_size, args.output_raster, args.train_ratio, args.n_estimators, args.max_depth, args.max_num_of_samples_per_class)




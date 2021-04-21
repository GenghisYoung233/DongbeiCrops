import glob, math, sys, os, time, random, argparse
import fiona
import rasterio as rio
import rasterio.mask
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm


def main(rasters_folder, stacked_raster, train_feature, output_dir, train_ratio, predict_ratio, pixel_ratio, cache):
    os.environ["GDAL_CACHEMAX"] = cache
    # stacking rasters if rasters_folder is provided.
    if rasters_folder:
        # Stacking time series rasters together
        rasters = glob.glob(os.path.join(rasters_folder, "*.tif")) + glob.glob(
            os.path.join(rasters_folder, "*.tiff"))
        stack(rasters, stacked_raster)

    # Generatin train/test datasets
    train_csv = train_feature.replace(".shp", "_train_datasat.csv")
    test_csv = train_feature.replace(".shp", "_test_datasat.csv")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_csv = os.path.join(output_dir, os.path.basename(train_csv))
        test_csv = os.path.join(output_dir, os.path.basename(test_csv))

    train_list, test_list, predict_list = split_train_feature(train_feature, train_ratio, predict_ratio)
    generate_training_data(stacked_raster, train_feature, test_list, test_csv, pixel_ratio)
    generate_training_data(stacked_raster, train_feature, train_list, train_csv, pixel_ratio)


def stack(rasters, out_raster):
    # Read in metadata
    first_raster = rio.open(rasters[0], 'r')
    out_raster = rio.open(out_raster, 'w', nodata=first_raster.nodata, driver='GTiff', height=first_raster.shape[0], width=first_raster.shape[1], count=first_raster.count*len(rasters), dtype=first_raster.read(1).dtype, crs=first_raster.crs, transform=first_raster.transform)
    band_index = 0
    for raster in tqdm(rasters):
        for band in rio.open(raster, "r").read():
            band_index+=1
            out_raster.write(band, band_index)
    out_raster.close()


def interpolation(input_raster, output_raster, n_channels, threshold, cache="50%"):
    os.environ["GDAL_CACHEMAX"] = cache
    src = rio.open(input_raster, 'r')
    meta = src.meta.copy()
    nodata=meta["nodata"]
    height=meta["height"]
    width=meta["width"]
    count=meta["count"]
    dtype = meta["dtype"]
    dst = rio.open(output_raster, 'w', nodata=nodata, driver='GTiff',
                               height=height, width=width,
                               count=threshold*n_channels, dtype=dtype,
                               crs=meta["crs"], transform=meta["transform"])
    # Read raster as array [bands, height, width]
    array = src.read()
    # Reshape array
    n_pixels = height*width
    array = array.transpose(1, 2, 0).reshape(n_pixels, int(count / n_channels), n_channels)
    # Loop through every pixel
    out_array = np.zeros((n_pixels, threshold, n_channels), dtype=dtype)
    for i, arr in tqdm(enumerate(array), total=n_pixels):
        # Delete nodata sequence
        nodata_seq = np.unique(np.where(arr == nodata)[0])
        arr = np.delete(arr, nodata_seq, axis=0)
        # Define a sequencelength threshold,
        # if sequencelength of pixel >= threshold, randomly extract from sequencelength
        # if sequencelength of 0.5*threshold < pixel < threshold, we randomly interpolate value from sequencelength to it
        # if sequencelength of pixel < 0.5*threshold, turn whole sequence to nodata
        sequencelength = arr.shape[0]
        if sequencelength >= threshold:
            idxs = np.random.choice(sequencelength, threshold, replace=False)
            idxs.sort()
            arr = arr[idxs]
        elif 0.5*threshold < sequencelength < threshold:
            idxs = np.random.choice(sequencelength, threshold - sequencelength, replace=False)
            idxs = np.append(np.arange(sequencelength), idxs)
            idxs.sort()
            arr = arr[idxs]
        else:
            arr = np.full((threshold, n_channels), nodata)

        # Write in pixel
        out_array[i, :] = arr

    # Reshape to [bands, width, height]
    out_array = out_array.reshape(height, width, -1).transpose(2, 0, 1)
    # Write out_array to out_raster
    dst.write(out_array)



def split_train_feature(train_feature, train_ratio=0.9, predict_ratio=0):
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
        train = fid_list[:int(len(fid_list)*train_ratio)]
        test = fid_list[int(len(fid_list)*train_ratio):int(len(fid_list)*(1-predict_ratio))]
        predict = fid_list[int(len(fid_list)*(1-predict_ratio)):int(len(fid_list)*1)]
        [train_list.append(i) for i in train]
        [test_list.append(i) for i in list(test)]
        [predict_list.append(i) for i in list(predict)]
    return train_list, test_list, predict_list


def generate_training_data(input_raster, train_feature, sample_list, out_csv, pixel_ratio=0.2):
    # read input_raster and train_feature
    src = rio.open(input_raster)
    shp = fiona.open(train_feature, "r")
    # Reproject train_feature if the crs are different
    if src.crs != shp.crs:
        shp = gpd.read_file(train_feature)
        shp = shp.to_crs(f"{src.crs}")
        train_feature = train_feature.replace(".shp", "_reprojected.shp")
        shp.to_file(train_feature)
        shp = fiona.open(train_feature, "r")

    # generate empty array to store data
    out_images = np.zeros(shape=(int(src.width*src.height/3), src.count+1), dtype=np.int64)
    
    row0 = 0
    # loop through every feature(polygon) to extract pixels' value within it.
    with tqdm(sample_list, desc = "Generating test/train datasets") as samples:
        for i in samples:
            # Masking raster using feature
            shape = [shp[i]["geometry"]]
            out_image, out_transform = rio.mask.mask(src, shape, crop=True)
            # Reshape array to (n_pixels, n_bands)
            out_image = out_image.reshape(-1).reshape(-1, src.count, order='F')
            # Delete nodata rows
            nodata_row = np.unique(np.where(out_image == -9999)[0])
            out_image = np.delete(out_image, nodata_row, axis=0)
            # Add classvalue to every row
            classvalue = int(shp[i]['properties']['Classvalue'])
            lc_id = np.linspace(classvalue, classvalue, out_image.shape[0], dtype="int64").reshape(-1, 1)
            out_image = np.append(lc_id, out_image, 1)
            # Write in out_image
            row1 = row0 + out_image.shape[0]
            out_images[row0:row1, :] = out_image
            row0 += out_image.shape[0]
    
    # Delete zero_rows
    zero_rows = np.where((out_images[:, 1:] == 0).all(axis=1))[0]
    out_images = np.delete(out_images, zero_rows, axis=0)
    # To avoid similarity, only part of the whole data are exported
    # Indices's datatype should be int32 cause maximum value of int16 is 32767
    indices = np.floor(np.arange(1, out_images.shape[0], 1/pixel_ratio)).astype('int64')
    out_images = np.take(out_images, indices, axis=0)
    # Add index in second column
    index = np.arange(0, out_images.shape[0], 1, dtype=np.int64)
    out_images = np.insert(out_images, 1, index, axis=1)
    # Export to .csv format
    df = pd.DataFrame(out_images, index)
    df.to_csv(out_csv, index=False, header=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating training datasets')
    parser.add_argument('--rasters_folder', dest='rasters_folder',
                        help='rasters folder',
                        default=None)
    parser.add_argument('--stacked_raster', dest='stacked_raster',
                        help='path for stacked raster',
                        default=None)
    parser.add_argument('--train_feature', dest='train_feature',
                        help='path for train feature',
                        default=None)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='path to store datasets',
                        default=None)
    parser.add_argument('--train_ratio', dest='train_ratio', type=float,
                        help='ratio of train part', default=0.7)
    parser.add_argument('--predict_ratio', dest='predict_ratio', type=float,
                        help='ratio of predict part, for TF/Keras only', default=0)
    parser.add_argument('--pixel_ratio', dest='pixel_ratio', type=float,
                        help='ratio of extracted pixel values to keep', default=0.5)
    parser.add_argument('--cache', dest='cache',
                        help='Set GDAL raster block cache size, may speed up processing with higher percentage, default is 5% of usable physical RAM',
                        type=str, default='20%')
    args = parser.parse_args()

    main(args.rasters_folder, args.stacked_raster, args.train_feature, args.output_dir, args.train_ratio, args.predict_ratio, args.pixel_ratio, args.cache)
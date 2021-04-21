## **Overview**

The **DongbeiCrops** contains deep learning models forked from [BreizhCrops](https://github.com/dl4sits/BreizhCrops) and several implementation to process your **custom datasets**, e.g., Gaofen satllite data in Northeast China (Dōngběi in Mandarin).

## **Installation**
Those step are for your a workstation on Windows 10 using miniconda.
Set and activate your python environment with the following commands:  
```powershell
conda config --add channels conda-forge
conda create -n dbc python=3.8 && conda activate dbc
conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.2 -c pytorch
conda install rasterio=1.2.0 geopandas jupyterlab=2.2.9 scikit-learn psutil tqdm PySimpleGUI
```

## **Running DongbeiCrops**
See [demo](https://github.com/GenghisYoung233/DongbeiCrops/blob/master/demo.ipynb) and [sample](https://github.com/GenghisYoung233/DongbeiCrops/tree/master/sample) to start a simple workflow.

# Benchmarking

## Table of contents

 1. [General Information](#general-information)
 2. [TrackMate](#trackmate)
 3. [SpotLearn](#spotlearn)
 4. [DetNet](#detnet)


## General Information

Benchmarking of deepBlink against comparable methods. In all cases, the methods were allowed to train / be optimized on the training and validation dataset while being evaluated on the test dataset. The mentioned scripts are available in the directory with corresponding names. Please run all scripts from inside the directories. In all subsequent benchmarks, the terminology is as follows:
* `DATABASE(S)`: the path to the npz dataset file
* `MODEL(S)`: the path to the pre-trained h5 model file
* `OUTPUT/SAVEDIR`: the directory into which output will be saved into (the structure is: OUTPUT/DATABASE_name/sub_directories)


## TrackMate

> Tinevez, JY.; Perry, N. & Schindelin, J. et al. (2017), "TrackMate: An open and extensible platform for single-particle tracking.", Methods 115: 80-90, PMID 27713081

TrackMate was originally written in Java but is available in Fiji and can be accessed through Fiji's python API. This, however, poses some limitations as only python's standard library is available in Jython. Therefore, significant amounts of file manipulations are required to convert the `npz` formatted dataset into individual, Fiji-readable files. The following steps will be performed to ensure an even playing ground between TrackMate and deepBlink:

1. **Preparation**: Save the default dataset format (npz) as individual image and label files – both train and valid will be combined for "training". To avoid clustering this folder with too many files, specify a `OUTPUT` directory into which all files will be saved.
    ```bash
    python trackmate_prepare.py --dataset DATASET --output OUTPUT
    ```
2. **Training**: In Fiji using TrackMate's python API, an extensive grid search will be performed over the following parameters. Note that training/testing was performed on a Linux machine. If you want to replicate the results on other systems, please change the Fiji launcher accordingly. The output is a list of CSV files containing the spot coordinates and an associated "quality" score. `BASEDIR` refers to the - during preparation created - folder in `OUTPUT` named like the `DATASET`.
    * Images
    * Detector method
    * Radius 
    ```bash
    Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_train_fiji.py 'basedir="BASEDIR"'
    ```
4. **Training - thresholding and metrics**: Back in python, various thresholds will be set on the spot's quality scores. On a per-image basis, the F1 scores will be computed.
    ```bash
    python trackmate_train.py --basedir BASEDIR 
    ```
5. **Apply to test set**: The best performing set of parameters (which have been saved to the file `BASEDIR/parameters.csv`) will be used and applied to the test dataset.
    ```bash
    Fiji.app/ImageJ-linux64 --ij2 --headless --run trackmate_test_fiji.py \
    'basedir="BASEDIR",detector="DETECTOR",radius="RADIUS",median="False"'
    ```
6. **Compute metrics**: Lastly apply the thresholds on the just computed spot lists.
    ```bash
    python trackmate_test.py --basedir BASEDIR --threshold THRESHOLD
    ```


## SpotLearn

> Gudla PR, Nakayama K, Pegoraro G, Misteli T. SpotLearn: Convolutional Neural Network for Detection of Fluorescence In Situ Hybridization (FISH) Signals in High-Throughput Imaging Approaches. Cold Spring Harb Symp Quant Biol. 2017;82:57-70. DOI:10.1101/sqb.2017.82.033761

Code used from the supplementary material of the publication on [GitHub](github.com/CBIIT/Misteli-Lab-CCR-NCI/tree/master/Gudla_CSH_2017/CNN_Python_Keras_TF/). The only changes were to the overall structure and image loading to accommodate for other datasets. Values, architecture, and training regiment were not altered. One key difference between SpotLearn and deepBlink is that SpotLearn outputs segmentation maps. This leads to two issues which were solved in the following way:
* Coordinates &rarr; training segmentation map: Differently sized and shaped "spot-like-objects" (diamonds or discs) placed on an empty canvas. To find the best size and shape, these hyperparameters were optimized.
* Output segmentation map &rarr; coordinates for benchmarking: Thresholding of probability distribution, labeling of spot regions, centroid coordinate output.

**Training**: Hyperparameter optimization on spot size and type using on one dataset. Returns the best model based on the validation score.
```bash
python spotlearn_train.py --dataset DATASET --savedir SAVEDIR
```

**Testing**: One off script on all datasets and models in parallel.
```bash
python spotlearn_test.py --datasets DATASETS --models MODELS
```


## DetNet

> T. Wollmann, C. Ritter, J. N. Dohrke, J.-Y. Lee, R. Bartenschlager, K. Rohr: DetNet: Deep Neural Network for Particle Detection in Fluorescence Microscopy Images. 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019); Venice, Italy, April 8-11, 2019

No source code was available in the original publication. The network and parameters were chosen based on the publication and commonly used values. The sigmoid shift alpha and batch sizes were tuned with 20 and 2 values respectively. Similar to SpotLearn, the output is also a segmentation map. Here, however, only a single pixel was labeled that corresponded to the ground truth coordinate.

**Training**: Hyperparameter optimization on alpha and batch size using one dataset. Returns the best model based on the validation score.
```bash
python detnet_train.py --dataset DATASET --savedir SAVEDIR
```

**Testing**: One off script on all datasets and models in parallel.
```bash
python detnet_test.py --datasets DATASETS --models MODELS
```

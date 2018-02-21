# Tensorflow - Semantic Segmentation Neural Network

This is an implementation of Convolutional Neural Network for Semantic Segmentation task.

This is a deep network of my own design named KNetS inspired by the [DeepLab v.2 Network] (https://arxiv.org/abs/1606.00915)

## Requirements

In addition to Python 3, Tensorflow Version>= 1.3, and Matlab the following packages are required:

numpy
scipy
pillow
matplotlib

These packages can be installed by running `pip3 install -r requirements.txt` or `pip3 install numpy scipy pillow matplotlib`.

## Usage

Currently, the code to generate the data was written to work only with the Camvid dataset.

### Step 1: Creating .mat file from raw data
Download [Camvid dataset] (http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

Put images data to the folder `./matlab/Camvid/701_StillsRaw_full`.

Put labels data to the folder `./matlab/Camvid/701_Label`.

Use Maltab to run `make_mat_files.m` in the `matlab` folder to create .mat data files.

Modify the `img_size` and partition rate if needed.

### Step 2: Creating tfrecord file from .mat file
Run `python3 data_utils.py` to create tfrecord files.

If needed, the user can use the `showing_data_from_tfrecord()` function in the `data_utils.py` to visualize the generated data in tfrecord files.

### Step 3: Training the network
Run `python3 main` to train the network.

Modify `batch_size`, `learning_rate` and `num_record` if needed.

After running the `saved_model` folder will be created.

Monitor the learning process by using terminal and navigate to the `graph` folder inside the `saved_model` folder, then run `tensorboard --logdir=KNetS` to activate Tensorboard. 

The training process can be resume if interrupted if the saved model has been saved. Just run `python3 main` again.

### Step 4: Testing the network
Put testing images in a folder and use the function `test_segmentize()` in the `data_utils.py` to segmentize the test samples

## Model

- KNetS
- FCN8

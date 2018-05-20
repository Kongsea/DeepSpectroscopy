# DeepSpectroscopy
Spectroscopy with Deep Learning

Analyzing spectrum with deep learning.

## Introduction

In recent years, deep learning has attracted an incresing attention in a wide range of research areas. However, as far as we know, there was no application of deep learning in the field of spectroscopy. So I would like to give an example to demonstrate the usage of deep learning to do qualitative and quantitative analysis of spectral data.

We use Laser Induced Breakdown Spectroscopy (LIBS) to illustrate the whole procecedure.

The spectroscopy files are organized as:

- data/1

       /1
       /2
       /3
       ...

- data/2

       /1
       /2
       /3
       ...

- data/3

       /1
       /2
       /3
       ...

- data/4

       /1
       /2
       /3
       ...

There are four classes of samples in all which are corresponding to the subfolders 1, 2, 3 and 4.

## Usage

### Requirements

- software

Requirements for Tensorflow (see: Tensorflow)

Python packages: numpy, csv, matplotlib

Only test on Ubuntu 16.04, Python 2.7 with TensorFlow 1.5.0

### Run:

Open Terminal in the root directory.

- ./scripts/process_data.sh

  to prepare the data for training model.
  The data were split to train, validation and test datasets respectively.

- ./scripts/train_qualitative.sh 1

  to train the qualitative model.

- ./scripts/train_quantitative.sh 1

  to train the quantitative model.

The logs will be saved to log folder.

# CTCSpeechRecognition

Work in progress. Implementation of the [Baidu Warp-CTC](https://github.com/baidu-research/warp-ctc) using torch7.
Creates a network based on the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) architecture using the Torch7 library, trained with the CTC activation function.

## Branches

There are currently two branches, master and voxforge:
* Master: This branch trains a neural network based on the [AN4 Audio database](http://www.speech.cs.cmu.edu/databases/an4/) of roughly 900 samples. Also included is a evaluation script which calculates the WER using the AN4 test dataset.
This branch is useful for understanding how the CTC Speech System works and is very easy to run after installation. Highly recommended to checkout this branch.
* Voxforge: This branch is like the above except it uses the [Voxforge Speech Corpus](www.voxforge.org) containing roughly 100k samples. This branch is far from complete and will represent a WORKING production model.
Once complete I will also release the fully trained model as pre-processing the dataset takes time.

## TODO
* Figure out a method to batch together sentences of similar length with appropriate padding and be able to utitilise the varying length sentences computation supported by warp-CTC
* Obtain a WER on the AN4 dataset around 13 using the basic spell checker. A suitable accuracy to move forward with scaling the project
* Create a Seq2Seq Attention based spell checker trained on the [Google Billion Words](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41880.pdf) dataset

To train the network using the AN4 dataset we use the AN4CTCTrain script and the parameters configured in the script.

To test the network we use the AN4CTCTest script, which generates the Word Error Rate(WER) based on the AN4 test dataset and our trained model.

## Installation

Follow instructions [here](https://github.com/SeanNaren/CTCSpeechRecognition/blob/master/INSTALL.md).

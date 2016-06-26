# CTCSpeechRecognition

[![Build Status](https://travis-ci.org/SeanNaren/CTCSpeechRecognition.svg?branch=master)](https://travis-ci.org/SeanNaren/CTCSpeechRecognition)

Implementation of [Baidu Warp-CTC](https://github.com/baidu-research/warp-ctc) using torch7.
Creates a network based on the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) architecture using the Torch7 library, trained with the CTC activation function.

## Features
* Train large models with large datasets via online loading using [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) and multi-GPU support.
* Supports variable length batches via masking.
* Implements the AN4 Audio database as an example of how a dataset can implemented.

## Branches

There are currently two branches, Master and AN4Phoneme:
* Master: This branch trains a neural network based on the [AN4 Audio database](http://www.speech.cs.cmu.edu/databases/an4/) of roughly 900 samples. Also included is an evaluation script which calculates the WER using the AN4 test dataset.
This branch is useful for understanding how the CTC Speech System works and is very easy to run after installation. Highly recommended to checkout this branch.
* AN4Phoneme: This branch is experimental and uses phonemes rather than character based predictions. This is fully credited and extended by [CCorfield](https://github.com/CCorfield) and his awesome work in porting to use phonemes.

## TODO
* Finish documentation for technical and general.
* Fix multi-GPU support by correctly handling the merging of multiple grads.

## Installation/Documentation

Follow Instructions/Documentation found in the wiki [here](https://github.com/SeanNaren/CTCSpeechRecognition/wiki/Installation) to set up and run the code.

Technical documentation can be found [here](http://ctcspeechrecognition.readthedocs.io/en/latest/).

## Acknowledgements

Lots of people helped/contributed to this project that deserve recognition:
* Soumith Chintala for his support on Torch7 and the vast open source projects he has contributed that made this project possible!
* Charles Corfield for his work on the Phoneme Dataset and his overall contribution and aid throughout.
* Will Frey for his thorough communication and aid in the development process.
* Ding Ling, Yuan Yang and Yan Xia for their significant contribution to online training, multi-gpu support and many other important features.
* Erich Elsen and the team from Baidu for their contribution of Warp-CTC that made this possible, and the encouraging words and support given throughout the project.

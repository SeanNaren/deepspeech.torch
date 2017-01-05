# deepspeech.torch

[![Build Status](https://travis-ci.org/SeanNaren/deepspeech.torch.svg?branch=master)](https://travis-ci.org/SeanNaren/deepspeech.torch)
[![Documentation Status](https://readthedocs.org/projects/ctcspeechrecognition/badge/?version=latest)](http://ctcspeechrecognition.readthedocs.io/en/latest/?badge=latest)


Implementation of [Baidu Warp-CTC](https://github.com/baidu-research/warp-ctc) using torch7.
Creates a network based on the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) architecture using the Torch7 library, trained with the CTC activation function.

## Features
* Train large models with large datasets via online loading using [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) and multi-GPU support.
* Supports variable length batches via padding.
* Implements the [AN4 Audio database](http://www.speech.cs.cmu.edu/databases/an4/) (50 mins of data).
Has also been extended to train using the [LibriSpeech](http://www.openslr.org/12/) dataset (1000 hours of data). Custom dataset preparation is explained in documentation.

## Branches

There are currently two branches, Master and Phoneme:
* Master: This branch trains DeepSpeech2. Also included is an evaluation script which calculates the WER/CER, as well as a prediction script.
This branch is useful for understanding how the DeepSpeech and CTC works and is easy to run after installation. Highly recommended to checkout this branch.
* Phonemes: This branch is experimental and uses phonemes rather than character based predictions. This is fully credited and extended by [CCorfield](https://github.com/CCorfield) and his awesome work in porting to use phonemes. In addition to this
I'd like to also thank [Shane Walker](https://github.com/walkers-mv) for his awesome recent conversion to use phonemes as well.

## Installation/Data Preparation/Documentation

Follow Instructions/Data Preparation/Documentation found in the wiki [here](https://github.com/SeanNaren/deepspeech.torch/wiki/Installation) to set up and run the code.

Technical documentation can be found [here](http://ctcspeechrecognition.readthedocs.io/en/latest/).

## Pre-trained Networks

Pre-trained networks are available for AN4 as well as LibriSpeech for CUDA only (since they use cudnn RNNs). Download Links and accuracies are below. DeepSpeech-light is a smaller model which is less intensive to train (based on LSTMs rather than RNNs).

### AN4

**an4Test**

|Network                | WER       | CER       |Link       |
|-----------------|:--------:|:--------:|:--------:|
|DeepSpeech-light| N/A     | N/A | N/A |
|DeepSpeech | 12    | 3.07 | [Download](https://github.com/SeanNaren/deepspeech.torch/releases/download/v1.0/an4_deepspeech.t7) |

### LibriSpeech

**Librispeech-test-clean**

|Network                | WER       | CER       |Link       |
|-----------------|:--------:|:--------:|:--------:|
|DeepSpeech-light| 15     | 1.34 | [Download](https://github.com/SeanNaren/deepspeech.torch/releases/download/v1.0/libri_deepspeech-light.t7) |
|DeepSpeech | 12    | 1.55 | [Download](https://github.com/SeanNaren/deepspeech.torch/releases/download/v1.0/libri_deepspeech.t7) |

**Librispeech-test-other**

|Network                | WER       | CER       |Link       |
|-----------------|:--------:|:--------:|:--------:|
|DeepSpeech-light| 36    | 3.80 | (Download Above) |
|DeepSpeech | 33    | 3.24 | (Download Above) |

Once you're set up, you can start training from these nets by using the below parameters (you might need to change the other parameters described in the wiki) after setting the project up:

```lua
th Train.lua -loadModel -loadPath /path/to/model.t7
```

## Acknowledgements

Lots of people helped/contributed to this project that deserve recognition:
* Soumith Chintala for his support on Torch7 and the vast open source projects he has contributed that made this project possible!
* Charles Corfield for his work on the Phoneme Dataset and his overall contribution and aid throughout.
* Will Frey for his thorough communication and aid in the development process.
* Ding Ling, Yuan Yang and Yan Xia for their significant contribution to online training, multi-gpu support and many other important features.
* Erich Elsen and the team from Baidu for their contribution of Warp-CTC that made this possible, and the encouraging words and support given throughout the project.
* Maciej Korzepa for his huge help in training a model on Librispeech!

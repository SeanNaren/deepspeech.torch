# CTCSpeechRecognition

Work in progress. Implementation of the <a href="https://github.com/baidu-research/warp-ctc/">Baidu Warp-CTC</a> using torch7. Feeds spectrogram data into a neural network using the Torch7 library, training itself with the CTC activation function.

Current implementation is CPU only but can be modified to support the GPU.

To install torch7 follow the guide <a href="http://torch.ch/docs/getting-started.html">here</a>.

To install the Baidu warp-ctc library follow the guide at the end of the readme <a href="https://github.com/baidu-research/warp-ctc/README.md">here</a> .

Other Dependencies can be installed via luarocks:
<a href="https://github.com/soumith/lua---audio">Audio Library for Torch</a>:
```
luarocks install https://raw.githubusercontent.com/soumith/lua---audio/master/audio-0.1-0.rockspec
```

<a href="https://github.com/torch/optim">Optim: numeric optimization package for Torch.</a>:
```
luarocks install optim
```

<a href="https://github.com/Element-Research/rnn">rnn: Recurrent Neural Network library for Torch7's nn.</a>:
```
luarocks install rnn
```

It is also suggested to update the following libraries:
```
luarocks install torch
luarocks install nn
luarocks install dpnn
```
For CUDA implementation:
```
luarocks install cutorch
luarocks install cunn
luarocks install cunnx
```
Main method located at AN4Test.lua.

Training data is currently the <a href="http://www.speech.cs.cmu.edu/databases/an4/">AN4 Audio database</a>. 

Within the AN4Test we specify the file path to the an4 dataset which can be downloaded <a href="http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz">here</a>. 

Extract the folder and give the filepath in the AN4Test.lua script. 

**Note that the data has to be converted into wav format, a bash script ConvertAN4ToWav.sh has been included to help with this (place this into the an4 directory then run).**

The current network is as follows (the input to the network is numberOfTimeFrames x frequencies where frequencies is 251).

```
nn.Sequential {
  (1): nn.Sequencer @ nn.Recursor @ nn.TemporalConvolution
  (2): nn.Sequencer @ nn.Recursor @ nn.ReLU
  (3): nn.Sequencer @ nn.Recursor @ nn.TemporalMaxPooling
  (4): nn.Sequencer @ nn.Recursor @ nn.ReLU
  (5): nn.Sequencer @ nn.Recursor @ nn.TemporalConvolution
  (6): nn.Sequencer @ nn.Recursor @ nn.ReLU
  (7): nn.Sequencer @ nn.Recursor @ nn.TemporalMaxPooling
  (8): nn.Sequencer @ nn.Recursor @ nn.TemporalConvolution
  (9): nn.Sequencer @ nn.Recursor @ nn.ReLU
  (10): nn.Sequencer @ nn.Recursor @ nn.Linear(450 -> 300)
  (11): nn.Sequencer @ nn.Recursor @ nn.ReLU
  (12): nn.BiSequencer @ nn.Sequential {
    [input -> (1) -> (2) -> (3) -> output]
    (1): nn.ConcatTable {
      input
        |`-> (1): nn.Sequencer @ nn.Recursor @ nn.Sequential {
        |      [input -> (1) -> (2) -> (3) -> output]
        |      (1): nn.FastLSTM(300 -> 200)
        |      (2): nn.FastLSTM(200 -> 100)
        |      (3): nn.FastLSTM(100 -> 100)
        |    }
        |`-> (2): nn.Sequential {
        |      [input -> (1) -> (2) -> (3) -> output]
        |      (1): nn.ReverseTable
        |      (2): nn.Sequencer @ nn.Recursor @ nn.Sequential {
        |        [input -> (1) -> (2) -> (3) -> output]
        |        (1): nn.FastLSTM(300 -> 200)
        |        (2): nn.FastLSTM(200 -> 100)
        |        (3): nn.FastLSTM(100 -> 100)
        |      }
        |      (3): nn.ReverseTable
        |    }
         ... -> output
    }
    (2): nn.ZipTable
    (3): nn.Sequencer @ nn.Recursor @ nn.JoinTable
  }
  (13): nn.Sequencer @ nn.Recursor @ nn.Linear(200 -> 27)
  (14): nn.Sequencer @ nn.Recursor @ nn.SoftMax
}

```

# CTCNetworkClient

Implementation of the <a href="https://github.com/baidu-research/warp-ctc/">Baidu Warp-CTC</a> using torch7.

Work in progress.

Main method located at AN4Test.lua.

The current network is as follows (the input to the network is numberOfTimeFrames x frequencies where frequencies is 251).

```
    nn.Sequencer(nn.TemporalConvolution(251,251,5,1)))
    nn.Sequencer(nn.ReLU()))
    nn.Sequencer(nn.TemporalMaxPooling(2,2)))
    nn.Sequencer(nn.TemporalConvolution(251,251,5,1)))
    nn.Sequencer(nn.ReLU()))
    nn.Sequencer(nn.TemporalConvolution(251,251,5,1)))
    nn.Sequencer(nn.ReLU()))
    nn.Sequencer(nn.BatchNormalization(251)))
    nn.Sequencer(nn.Linear(251,251)))
    nn.Sequencer(nn.ReLU()))
    nn.BiSequencer(nn.FastLSTM(251,40),nn.FastLSTM(251,40)))
    nn.Sequencer(nn.BatchNormalization(40*2)))
    nn.BiSequencer(nn.FastLSTM(40*2,30),nn.FastLSTM(40*2,30)))
    nn.Sequencer(nn.BatchNormalization(30*2)))
    nn.BiSequencer(nn.FastLSTM(30*2,20),nn.FastLSTM(30*2,20)))
    nn.Sequencer(nn.BatchNormalization(20*2)))
    nn.Sequencer(nn.Linear(20*2,27)))
    nn.Sequencer(nn.SoftMax()))
```

# CTCNetworkClient

Implementation of the <a href="https://github.com/baidu-research/warp-ctc/">Baidu Warp-CTC</a> using torch7.

Work in progress.

Main method located at AN4Test.lua.

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

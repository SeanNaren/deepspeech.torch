# DeepSpeechModel

Defines the deep speech 2 conv+rnn architecture.

### deepSpeech(nGPU)

Defines the torch architecture for Deep Speech 2 as a function that can be called. Returns the final model

`nGPU` Number of GPUs that will be used.

Also defined in this class are a few variables that can be modified if so chosen:

```lua
local rnnHiddenSize = 700 -- size of each rnn hidden layers (rnnHiddenSize inputDim, rnnHiddenSize outputDim)
local nbOfHiddenLayers = 7  -- number of hidden RNN layers to add
```

### calculateInputSizes(sizes)

A function that calculates the sequence sizes after the convolutional layers. Used in the loss calculations in CTC, so the network isn't
penalised for the padded sequences. Returns a same sized tensor.

`sizes` Real size of each sentence in the training sample as a 1D tensor.
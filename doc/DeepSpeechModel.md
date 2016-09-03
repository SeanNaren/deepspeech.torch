# DeepSpeechModel

Defines the deep speech 2 conv+rnn architecture.

### deepSpeech(opt)

Defines the torch architecture for Deep Speech 2 as a function that can be called. Returns the final model

`opt` Defines the options we use including using GPUS, hidden size and number of layers for the RNNs.

### calculateInputSizes(sizes)

A function that calculates the sequence sizes after the convolutional layers. Used in the loss calculations in CTC, so the network isn't
penalised for the padded sequences. Returns a same sized tensor.

`sizes` Real size of each sentence in the training sample as a 1D tensor.
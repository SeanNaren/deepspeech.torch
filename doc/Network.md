# Network

Handles interactions with the neural network for training and testing. Configured by network parameters given in
constructor.

### Network:init(networkParams)

Constructor of the Network class. Below defines each parameter that can be taken as input.

```lua
local networkParams = {
    loadModel = false, -- Set to true if loading a model into the Network class rather than training.
    saveModel = true, -- Set to true if saving the model after training.
    modelName = 'DeepSpeechModel', -- The name of the lua class containing the network architecture
    backend = 'cudnn', -- supports cudnn (GPU) and rnn (CPU)
    nGPU = 1, -- Number of GPUs, set -1 to use CPU
    trainingSetLMDBPath = './prepare_an4/train/', -- online loading path from the LMDB dataset for training.
    validationSetLMDBPath = './prepare_an4/test/', -- online loading path from the LMDB dataset for testing.
    logsTrainPath = './logs/TrainingLoss/', -- Where training logs will be stored.
    logsValidationPath = './logs/ValidationScores/', -- Where testing score logs will be stored.
    modelTrainingPath = './models/', -- Where models will be stored on saving.
    fileName = 'CTCNetwork.t7',
    dictionaryPath = './dictionary', -- Contains the alphabet/characters that we are to predict on.
    batchSize = 20, -- The sizes of batches that we are passing into the network in training.
    validationBatchSize = 1, -- Validation batch sizes (should be kept at 1, since we pass 1 sample at a time).
    validationIterations = 20, -- Number of validation iterations (kept small, because we only want to run a few tests per epoch).
    saveModelInTraining = false, -- saves model periodically through training
    saveModelIterations = 50 -- If saveModelInTraining set to true, we save every 50 epochs.
}
```

### Network:prepSpeechModel(networkParams)

Used to create the model via the defined modelName using the configured backend and numbner of GPUs.

### Network:testNetwork(epoch)

Tests the current stored model via the word error rate.

`epoch` can be used to detail the epoch number in the logs when testing scores are stored.

### Network:trainNetwork(epochs, sgd_params)

Trains a network stored in the `Network` class. Uses multiple threads in an online loading fashion to load the data from the disk.

`epochs` defines the number of iterations of training that will occur across the entire dataset (each epochs trains on the entire dataset).

`sgd_params` defines the SGD parameters for the optim library such as below.

```lua
local sgdParams = {
    learningRate = 5e-4,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}
```

### Network:createLossGraph()

After training, when called will use gnuplot (through wrapper in the optim library) to generate a graph of the loss and word error rate over epochs.

### Network:saveNetwork(saveName)

Will save the model currently stored in the network class to disk, at the pre-defined save location with the given `saveName`.

### Network:loadNetwork(saveName, modelName, is_cudnn)

Loads the network from the save location, stored using the pre-defined save name.

`saveName` The name as to which the network was saved as

`modelName` The name of the class that stores the model or architecture.

`is_cudnn` Defines if we should use cuDNN for GPU support.
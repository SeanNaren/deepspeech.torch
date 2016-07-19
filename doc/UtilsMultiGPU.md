# UtilsMultiGPU

Handles multi-gpu setups of the architecture.

### makeDataParallel(model, nGPU, is_cudnn)

Converts the model into a multi-gpu set up if necessary using DataParallelTable.

`model` The Torch network model to modify for configured GPUs.

`nGPU` Number of GPUs.

`is_cudnn` Set to true if using cuDNN backend.

### saveDataParallel(filename, model)

Saves the model to disk.

`fileName` Location to save the file.

`model` The Torch network model to save.

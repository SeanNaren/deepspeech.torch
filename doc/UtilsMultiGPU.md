# UtilsMultiGPU

Handles multi-gpu setups of the architecture.

### makeDataParallel(model, nGPU)

Converts the model into a multi-gpu set up if necessary using DataParallelTable.

`model` The Torch network model to modify for configured GPUs.

`nGPU` Number of GPUs.

### saveDataParallel(modelPath, model)

Saves the model to disk.

`modelPath` Location to save the model.

`model` The Torch network model to save.

### loadDataParallel(modelPath, nGPU)

Loads a model saved using the above methods.

`modelPath` Location to load the model.

`nGPU` Number of GPUs to load to.
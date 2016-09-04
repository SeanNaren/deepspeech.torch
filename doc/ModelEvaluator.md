# ModelEvaluator

Handles calculation of word error rate using an LMDB dataset. For more information on the calculation, see [Evaluator](https://github.com/SeanNaren/CTCSpeechRecognition/doc/Evaluator.md).

### ModelEvaluator:__init(isGPU, datasetPath, mapper, testBatchSize, logsPath)

'isGPU' Whether to use the GPU (CUDA) or CPU.

`datasetPath` the path to the LMDB test dataset to use in evaluation.

`mapper` Maps predicted numeric values to characters, see [Mapper](https://github.com/SeanNaren/CTCSpeechRecognition/doc/Mapper.md) for more details.

`testBatchSize` The size of the batches to pass the network.

`logsPath` File path to put the details of evaluations into.


### ModelEvaluator:runEvaluation(model, verbose, epoch)

Calculates the word error rate and character error rate averaged over the test iterations. Uses the same threading as the training process does to load batches from the dataset.

`model` The Torch model to evaluate.

`verbose` If set to true, will store details of WER calculations into the log files.

`epoch` Determines the epoch number that is written in the log files for this calculation.
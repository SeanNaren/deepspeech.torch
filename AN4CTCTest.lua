local Network = require 'Network'

-- Load the network from the saved model. Options can be overrided on command line run.
local cmd = torch.CmdLine()
cmd:option('-loadModel', true, 'Load previously saved model')
cmd:option('-saveModel', false, 'Save model after training/testing')
cmd:option('-modelName', 'DeepSpeechModel', 'Name of class containing architecture')
cmd:option('-nGPU', 1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-trainingSetLMDBPath', './prepare_datasets/an4_lmdb/train/', 'Path to LMDB training dataset')
cmd:option('-validationSetLMDBPath', './prepare_datasets/an4_lmdb/test/', 'Path to LMDB test dataset')
cmd:option('-logsTrainPath', './logs/TrainingLoss/', ' Path to save Training logs')
cmd:option('-logsValidationPath', './logs/ValidationScores/', ' Path to save Validation logs')
cmd:option('-modelPath', 'deepspeech.t7', 'Path of final model to save/load')
cmd:option('-dictionaryPath', './dictionary', ' File containing the dictionary to use')
cmd:option('-batchSize', 20, 'Batch size in training')
cmd:option('-validationBatchSize', 32, 'Batch size for validation')

local opt = cmd:parse(arg)
Network:init(opt)

print("Testing network...")
local wer, cer = Network:testNetwork()
print(string.format('Avg WER: %2.f  Avg CER: %.2f', 100 * wer, 100 * cer))
print(string.format('More information written to log file at %s', opt.logsValidationPath))

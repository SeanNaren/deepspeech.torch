local Network = require 'Network'

-- Options can be overrided on command line run.
local cmd = torch.CmdLine()
cmd:option('-loadModel', false, 'Load previously saved model')
cmd:option('-saveModel', true, 'Save model after training/testing')
cmd:option('-modelName', 'DeepSpeechModel', 'Name of class containing architecture')
cmd:option('-nGPU', 1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-trainingSetLMDBPath', './prepare_datasets/an4_lmdb/train/', 'Path to LMDB training dataset')
cmd:option('-validationSetLMDBPath', './prepare_datasets/an4_lmdb/test/', 'Path to LMDB test dataset')
cmd:option('-logsTrainPath', './logs/TrainingLoss/', ' Path to save Training logs')
cmd:option('-logsValidationPath', './logs/ValidationScores/', ' Path to save Validation logs')
cmd:option('-saveModelInTraining', false, 'save model periodically through training')
cmd:option('-modelTrainingPath', './models/', ' Path to save periodic training models')
cmd:option('-saveModelIterations', 50, 'When to save model through training')
cmd:option('-modelPath', 'deepspeech.t7', 'Path of final model to save/load')
cmd:option('-dictionaryPath', './dictionary', ' File containing the dictionary to use')
cmd:option('-epochs', 70, 'Number of epochs for training')
cmd:option('-learningRate', 3e-4, ' Training learning rate')
cmd:option('-learningRateAnnealing', 1.1, 'Factor to anneal lr every epoch')
cmd:option('-maxNorm', 400, 'Max norm used to normalize gradients')
cmd:option('-momentum', 0.90, 'Momentum for SGD')
cmd:option('-batchSize', 20, 'Batch size in training')
cmd:option('-permuteBatch', false, 'Set to true if you want to permute batches AFTER the first epoch')
cmd:option('-validationBatchSize', 20, 'Batch size for validation')
cmd:option('-LSTM', false, 'Use LSTMs rather than RNNs')
cmd:option('-hiddenSize', 1760, 'RNN hidden sizes')
cmd:option('-nbOfHiddenLayers', 7, 'Number of rnn layers')

local opt = cmd:parse(arg)

--Parameters for the stochastic gradient descent (using the optim library).
local optimParams = {
    learningRate = opt.learningRate,
    learningRateAnnealing = opt.learningRateAnnealing,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}

--Create and train the network based on the parameters and training data.
Network:init(opt)

Network:trainNetwork(opt.epochs, optimParams)

--Creates the loss plot.
Network:createLossGraph()
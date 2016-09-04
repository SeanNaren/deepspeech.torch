require 'nn'
require 'audio'
require 'Mapper'
require 'UtilsMultiGPU'
local cmd = torch.CmdLine()
cmd:option('-modelPath', 'deepspeech.t7', 'Path of model to load')
cmd:option('-audioPath', '', 'Path to the input audio to predict on')
cmd:option('-dictionaryPath', './dictionary', 'File containing the dictionary to use')
cmd:option('-windowSize', 0.02, 'Window Size of audio')
cmd:option('-stride', 0.01, 'Stride of audio')
cmd:option('-sampleRate', 16000, 'Rate of audio (default 16khz)')
cmd:option('-nGPU', 1)

local opt = cmd:parse(arg)

if opt.nGPU > 0 then
    require 'cunn'
    require 'cudnn'
    require 'BatchBRNNReLU'
end

local model =  loadDataParallel(opt.modelPath, opt.nGPU)
local mapper = Mapper(opt.dictionaryPath)

local wave = audio.load(opt.audioPath)
local spect = audio.spectrogram(wave, opt.windowSize * opt.sampleRate, 'hamming', opt.stride * opt.sampleRate):float() -- freq-by-frames tensor

-- normalize the data
local mean = spect:mean()
local std = spect:std()
spect:add(-mean)
spect:div(std)

spect = spect:view(1, 1, spect:size(1), spect:size(2))

if opt.nGPU > 0 then
    spect = spect:cuda()
    model = model:cuda()
end

model:evaluate()
local predictions = model:forward(spect)
local tokens = mapper:decodeOutput(predictions[1])
local text = mapper:tokensToText(tokens)

print(text)
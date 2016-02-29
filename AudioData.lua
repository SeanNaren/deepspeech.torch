--Retrieves audio datasets. Currently retrieves the AN4 dataset by giving the folder directory.
require 'lfs'
require 'audio'
cutorch = require 'cutorch'
local AudioData = {}
local alphabet = {
    '$', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' '
}

local alphabetMapping = {}
local indexMapping = {}
for index, character in ipairs(alphabet) do
    alphabetMapping[character] = index - 1
    indexMapping[index - 1] = character
end

--Given an index returns the letter at that index.
function AudioData.findLetter(index)
    return indexMapping[index]
end

function AudioData.retrieveAN4TrainingDataSet(folderDirPath, windowSize, stride, batchSize, noiseFilePath)
    local audioLocationPath = folderDirPath .. "/etc/an4_train.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_train.transcription"

    local targets = {}

    for line in io.lines(transcriptPath) do
        local label = {}
        for string in string.gmatch(line, ">([^<]*)<") do
            --This line removes the space at the beginning and end of the sentence input.
            string = string:sub(2):sub(1, -2)
            for i = 1, #string do
                local character = string:sub(i, i)
                table.insert(label, alphabetMapping[character])
            end
            table.insert(targets, label)
        end
    end
    local inputs, targets = an4Dataset(folderDirPath, audioLocationPath, windowSize, stride, batchSize, noiseFilePath, targets)
    return inputs, targets
end

function AudioData.retrieveAN4TestDataSet(folderDirPath, windowSize, stride, batchSize, noiseFilePath)
    local audioLocationPath = folderDirPath .. "/etc/an4_test.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_test.transcription"

    local targets = {}

    for line in io.lines(transcriptPath) do
        local label = {}
        local string = line:gsub('%b()', '')
        --Remove the space at the end of the line.
        string = string:sub(1, -2)
        for i = 1, #string do
            local character = string:sub(i, i)
            table.insert(label, alphabetMapping[character])
        end
        table.insert(targets, label)
    end
    local inputs, targets = an4Dataset(folderDirPath, audioLocationPath, windowSize, stride, batchSize, noiseFilePath, targets)
    return inputs, targets
end

function an4Dataset(folderDirPath, audioLocationPath, windowSize, stride, batchSize, noiseFilePath, targets)
    local inputs = {}
    local counter = 0
    for audioPath in io.lines(audioLocationPath) do
        counter = counter + 1
        local audioData = audio.load(folderDirPath .. "/wav/" .. audioPath .. ".wav")
        --We transpose the frequency/time to now put time on the x axis, frequency on the y axis.
        local spectrogram = audio.spectrogram(audioData, windowSize, 'hamming', stride):transpose(1, 2)
        table.insert(inputs, spectrogram)
        if (math.fmod(counter, 100) == 0) then print(counter, " completed") end
    end

    local noiseFile = audio.load(noiseFilePath)
    local noiseSpectrogram = audio.spectrogram(noiseFile, windowSize, 'hamming', stride):transpose(1, 2)
    local dataset = createDataSet(inputs, targets, batchSize, noiseSpectrogram)
    return dataset
end

--Creates the dataset depending on the batchSize given. We also pad all the inputs so they are the same size using the
--noise tensor to fill every tensor to the same size via padding.
function createDataSet(inputs, targets, batchSize, noiseTensor)
    local dataset = {}
    for t = 1, #inputs, batchSize do
        local inputsBatch = {}
        local targetsBatch = {}
        for i = t, math.min(t + batchSize - 1, #inputs) do
            table.insert(inputsBatch, inputs[i])
            table.insert(targetsBatch, targets[i])
        end
        table.insert(dataset, { padDataset(inputsBatch, noiseTensor), targetsBatch })
    end
    local pointer = 1
    function dataset:size() return #dataset end

    function dataset:nextData()
        local sample = dataset[pointer]
        pointer = pointer + 1
        if (pointer > dataset:size()) then pointer = 1 end
        return sample[1], sample[2]
    end

    return dataset
end

--Pads a dataset with 0's so all tensors are off the same size.
function padDataset(totalInput, noiseTensor)
    local allSizes, maxSize = findMaxSize(totalInput)
    local noiseTimeStepIterator = createNoiseIterator(noiseTensor)
    for i = 1, #totalInput do
        local input = torch.totable(totalInput[i])
        while (#input < maxSize) do
            table.insert(input, noiseTimeStepIterator:nextNoiseTimeStep())
        end
        --TODO here we have set the type to cuda, decide whether to hardcore gpu training.
        totalInput[i] = torch.Tensor(input):cuda()
    end
    return totalInput
end

function createNoiseIterator(noiseTensor)
    local noiseTensorTable = torch.totable(noiseTensor)
    local pointer = 1
    local noiseSize = #noiseTensorTable
    function noiseTensorTable:nextNoiseTimeStep()
        local noiseTimeStep = noiseTensorTable[pointer]
        pointer = pointer + 1
        if (pointer > noiseSize) then pointer = 1 end
        return noiseTimeStep
    end

    return noiseTensorTable
end

--Returns the largest tensor size and all sizes in a table of tensors
function findMaxSize(tensors)
    local maxSize = 0
    local allSizes = {}
    for i = 1, #tensors do
        local tensorSize = tensors[i]:size(1)
        if (tensorSize > maxSize) then maxSize = tensorSize end
        table.insert(allSizes, tensorSize)
    end
    return allSizes, maxSize
end


return AudioData
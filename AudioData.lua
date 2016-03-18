--Retrieves audio datasets. Currently retrieves the AN4 dataset by giving the folder directory.
require 'lfs'
require 'audio'
cutorch = require 'cutorch'
require 'xlua'
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

function AudioData.retrieveAN4TrainingDataSet(folderDirPath, windowSize, stride)
    local audioLocationPath = folderDirPath .. "/etc/an4_train.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_train.transcription"
    local nbSamples = 948 -- Amount of samples found in the AN4 training set.

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
    local inputs, targets = an4Dataset(folderDirPath, audioLocationPath, windowSize, stride, targets, nbSamples)
    return inputs, targets
end

function AudioData.retrieveAN4TestDataSet(folderDirPath, windowSize, stride)
    local audioLocationPath = folderDirPath .. "/etc/an4_test.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_test.transcription"
    local nbSamples = 130 -- Amount of samples found in the AN4 test set.
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
    local inputs, targets = an4Dataset(folderDirPath, audioLocationPath, windowSize, stride, targets, nbSamples)
    return inputs, targets
end

--Used by the Seq2Seq model. Retrieves both the test and training transcripts for AN4.
function AudioData.retrieveAN4TranscriptSet(folderDirPath)
    local transcriptPathTrain = folderDirPath .. "/etc/an4_train.transcription"
    local transcriptPathTest = folderDirPath .. "/etc/an4_test.transcription"

    local trainingTranscripts = {}
    local testTranscripts = {}

    local function convertToLabel(transcripts, line)
        local label = {}
        local string = line:gsub('%b()', '')
        --Remove the space at the end of the line.
        string = string:sub(1, -2)
        for i = 1, #string do
            local character = string:sub(i, i)
            table.insert(label, alphabetMapping[character])
        end
        table.insert(transcripts, label)
    end

    for line in io.lines(transcriptPathTrain) do
        convertToLabel(trainingTranscripts, line)
    end
    for line in io.lines(transcriptPathTest) do
        convertToLabel(testTranscripts, line)
    end
    return trainingTranscripts
end

function an4Dataset(folderDirPath, audioLocationPath, windowSize, stride, targets, nbOfSamples)
    local inputs = {}
    local counter = 0
    for audioPath in io.lines(audioLocationPath) do
        counter = counter + 1
        local audioData = audio.load(folderDirPath .. "/wav/" .. audioPath .. ".wav")
        -- We transpose the frequency/time to now put time on the x axis, frequency on the y axis.
        local spectrogram = audio.spectrogram(audioData, windowSize, 'hamming', stride):transpose(1, 2)
        table.insert(inputs, spectrogram)
        xlua.progress(counter,nbOfSamples)
    end
    local dataset = createDataSet(inputs, targets)
    return dataset
end

function createDataSet(inputs, targets)
    local dataset = {}
    for i = 1, #inputs do
        --Wrap the targets into a table (batches would go into the table, but no batching supported).
        --Convert the tensor into a cuda tensor for gpu calculations.
        table.insert(dataset, {inputs[i]:cuda(), {targets[i]}})
    end
    local pointer = 1
    function dataset:size() return #dataset end

    function dataset:nextData()
        local sample = dataset[pointer]
        pointer = pointer + 1
        if (pointer > #dataset) then pointer = 1 end
        return sample[1], sample[2]
    end

    return dataset
end


return AudioData
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

function AudioData.retrieveAN4TrainingDataSet(folderDirPath, windowSize, stride)
    local audioLocationPath = folderDirPath .. "/etc/an4_train.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_train.transcription"
    local inputs, targets = an4Dataset(folderDirPath, audioLocationPath, transcriptPath, windowSize, stride)
    return inputs, targets
end

function AudioData.retrieveAN4TestDataSet(folderDirPath, windowSize, stride)
    local audioLocationPath = folderDirPath .. "/etc/an4_test.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_test.transcription"
    local inputs, targets = an4Dataset(folderDirPath, audioLocationPath, transcriptPath, windowSize, stride)
    return inputs, targets
end

--Given an index returns the letter at that index.
function AudioData.findLetter(index)
    return indexMapping[index]
end

function an4Dataset(folderDirPath, audioLocationPath, transcriptPath, windowSize, stride)
    local inputs = {}
    local targets = {}
    local counter = 0
    for audioPath in io.lines(audioLocationPath) do
        counter = counter + 1
        local audioData = audio.load(folderDirPath .. "/wav/" .. audioPath .. ".wav")
        local spectrogram = audio.spectrogram(audioData, windowSize, 'hamming', stride)
        local transposedSpectrogram = spectrogram:transpose(1, 2)
        table.insert(inputs, transposedSpectrogram)
        if (math.fmod(counter, 100) == 0) then print(counter, " completed") end
    end
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
    return inputs, targets
end

return AudioData
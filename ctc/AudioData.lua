require'lfs'
require 'audio'
require 'image'
cutorch = require 'cutorch'
local AudioData = {}
local alphabet = {' ','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

local alphabetMapping = {}
local indexMapping = {}
for index,character in ipairs(alphabet) do
    alphabetMapping[character] = index - 1
    indexMapping[index - 1] = character
end

function AudioData.retrieveAN4TrainingDataSet(folderDirPath)
    local audioLocationPath = folderDirPath .. "/etc/an4_train.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_train.transcription"
    local inputs,targets = an4Dataset(folderDirPath,audioLocationPath,transcriptPath)
    return inputs,targets
end


function AudioData.retrieveAN4TestDataSet(folderDirPath)
    local audioLocationPath = folderDirPath .. "/etc/an4_test.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_test.transcription"
    local inputs,targets = an4Dataset(folderDirPath,audioLocationPath,transcriptPath)
    return inputs,targets
end

function an4Dataset(folderDirPath,audioLocationPath,transcriptPath)
    local inputs = {}
    local targets = {}
    local counter = 0
    local audioPaths = io.lines(audioLocationPath)
    for audioPath in audioPaths do
        counter = counter + 1
        local audioData = audio.load(folderDirPath .. "/wav/" .. audioPath .. ".wav")
        local spectrogram = audio.spectrogram(audioData, 500, 'hann', 50)
        local transposedSpectrogram = spectrogram:transpose(1,2)
        table.insert(inputs,transposedSpectrogram)
        if(math.fmod(counter,100) == 0) then print(counter," completed") end
    end
    for line in io.lines(transcriptPath) do
        local label = {}
        for character in string.gmatch(line, "([A-Z])") do
            table.insert(label,alphabetMapping[character])
        end
        table.insert(targets,label)
    end
    return inputs,targets
end
return AudioData

require'lfs'
require 'audio'
require 'image'
local AudioData = {}
local alphabet = {' ','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

local alphabetMapping = {}
local indexMapping = {}
for index,character in ipairs(alphabet) do
    alphabetMapping[character] = index - 1
    indexMapping[index - 1] = character
end

function AudioData.retrieveAN4DataSet(folderDirPath)
    local audioLocationPath = folderDirPath .. "/etc/an4_train.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_train.transcription"
    local inputs = {}
    local targets = {}
    local counter = 0
    for audioPath in io.lines(audioLocationPath) do
        counter = counter + 1
        local audioData = audio.load(folderDirPath .. "/wav/" .. audioPath .. ".wav")
        local spectrogram = audio.spectrogram(audioData, 500, 'hann', 50)
        local transposedSpectrogram = spectrogram:transpose(1,2)
        table.insert(inputs,transposedSpectrogram)
        if(counter == 10) then break end
    end
    counter = 0
    for line in io.lines(transcriptPath) do
        local label = {}
        counter = counter + 1
        for character in string.gmatch(line, "([A-Z])") do
            table.insert(label,alphabetMapping[character])
        end
        table.insert(targets,label)
        if(counter == 10) then break end
    end
    return inputs,targets
end
return AudioData

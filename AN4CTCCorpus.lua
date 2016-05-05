--Retrieves audio datasets. Currently retrieves the AN4 dataset by giving the folder directory.
require 'lfs'
require 'audio'
require 'xlua'

local AN4CTCCorpus = {}
local alphabet = {
    '$', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' '
}

local alphabetMapping = {}
local indexMapping = {}
for index, character in ipairs(alphabet) do
    alphabetMapping[character] = index - 1
    indexMapping[index - 1] = character
end

--Given an index returns the letter at that index.
function AN4CTCCorpus.findLetter(index)
    return indexMapping[index]
end

-- Local functions to help process corpus into {input, labels, text}
local function convertTextToLabels(text)
    local label = {}
    for i = 1, #text do
        local character = text:sub(i, i)
        table.insert(label, alphabetMapping[character])
    end
    return label
end

local function getDataSet(audioPath, transcriptPath, folderDirPath, windowSize, windowStride)
    local dataSet = {}

    -- Run through the transcript and generate truth text and labels (targets)
    local i = 1
    for line in io.lines(transcriptPath) do
        line = string.lower(line)

        -- Trim meta data so that we have the truth text
        local text = line:gsub('<s>', ''):gsub('</s>', ''):gsub('^%s', ''):gsub('%(.*%)', ''):gsub('%s*$', '')

        -- Convert the truth text to pronunciations
        local label = convertTextToLabels(text)

        -- Put what we have, thus far, into our dataset
        table.insert(dataSet, { text = text, label = label })

        i = i + 1
    end
    local dataSetSize = i - 1

    -- Run through file containing audio file locations
    i = 1
    for line in io.lines(audioPath) do
        local file = folderDirPath .. "/wav/" .. line .. ".wav"
        -- audio.load returns channel x frequency x time
        local sound = audio.load(file)
        -- audio.spectrogram returns f x t [which is the layout we need for spatial convolution]
        local spectrum = audio.spectrogram(sound, windowSize, 'hamming', windowStride)
        dataSet[i].input = spectrum
        xlua.progress(i, dataSetSize)
        i = i + 1
    end
    return dataSet
end


-- Public interface to return prepped data sets
function AN4CTCCorpus.getTrainingData(folderDirPath, windowSize, stride)
    local audioLocationPath = folderDirPath .. "/etc/an4_train.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_train.transcription"
    return getDataSet(audioLocationPath, transcriptPath, folderDirPath, windowSize, stride)
end

function AN4CTCCorpus.getTestingData(folderDirPath, windowSize, stride)
    local audioLocationPath = folderDirPath .. "/etc/an4_test.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_test.transcription"
    return getDataSet(audioLocationPath, transcriptPath, folderDirPath, windowSize, stride)
end


return AN4CTCCorpus
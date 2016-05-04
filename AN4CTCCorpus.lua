-- Process the AN4 Corpus into {truth text, input, labels}
-- In this case the labels are phonemes

local AN4CTCCorpus = {}

require 'lfs'
require 'audio'
require 'xlua'

local phones = {
    'sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g',
    'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th',
    'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'
}

local phoneToIndex = {}
local indexToPhone = {}
for index, phone in ipairs(phones) do
    phoneToIndex[phone] = index
    indexToPhone[index] = phone
end

function AN4CTCCorpus.phoneToIndex(char)
    return phoneToIndex[char]
end

function AN4CTCCorpus.indexToPhone(index)
    return indexToPhone[index]
end

-- Local functions to help process corpus into {input, labels, text}
local function convertTextToLabels(labels, text, AN4PhonemeDictionary)
    for word in string.gfind(text, '%S+') do
        local phones = AN4PhonemeDictionary.LookUpWord(word)
        if (phones ~= nil) then
            for phone in string.gfind(phones, '%S+') do
                table.insert(labels, phoneToIndex[phone])
            end
        end
    end
end

local function getDataSet(audioPath, transcriptPath, folderDirPath, windowSize, windowStride, AN4PhonemeDictionary)
    local dataSet = {}

    -- Run through the transcript and generate truth text and labels (targets)
    local i = 1
    for line in io.lines(transcriptPath) do
        line = string.lower(line)

        -- Trim meta data so that we have the truth text
        local text = line:gsub('<s>', ''):gsub('</s>', ''):gsub('^%s', ''):gsub('%(.*%)', ''):gsub('%s*$', '')

        -- Convert the truth text to pronunciations
        local labels = {}
        convertTextToLabels(labels, text, AN4PhonemeDictionary)

        -- Put what we have, thus far, into our dataset
        table.insert(dataSet, { text = text, labels = labels })

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
function AN4CTCCorpus.getTrainingData(folderDirPath, windowSize, stride, AN4PhonemeDictionary)
    local audioLocationPath = folderDirPath .. "/etc/an4_train.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_train.transcription"
    return getDataSet(audioLocationPath, transcriptPath, folderDirPath, windowSize, stride, AN4PhonemeDictionary)
end

function AN4CTCCorpus.getTestingData(folderDirPath, windowSize, stride, AN4PhonemeDictionary)
    local audioLocationPath = folderDirPath .. "/etc/an4_test.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_test.transcription"
    return getDataSet(audioLocationPath, transcriptPath, folderDirPath, windowSize, stride, AN4PhonemeDictionary)
end


return AN4CTCCorpus
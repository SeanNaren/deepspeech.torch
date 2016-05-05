--Retrieves audio datasets. Currently retrieves the AN4 dataset by giving the folder directory.
require 'lfs'
require 'audio'
require 'xlua'
local AudioData = {}
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
function AudioData.findLetter(index)
    return indexMapping[index]
end



local function convertLineToLabels(labels, transcripts, line)
    --[[
        input:
            line: ERASE C Q Q F SEVEN (id)
        output:
            label: {3,7,1,2,8}
            transcript: erase c q q f seven
    --]]

    local label = {}
    line = string.lower(line)
    -- Remove: beginning space, BOS, EOS, fileid, final space, string ids (<s> and </s>).
    line = line:gsub('^%s', ''):gsub('', ''):gsub('', ''):gsub('%(.+%)', ''):gsub('%s$', ''):gsub('<s>', ''):gsub('</s>', '')
    --Remove the space at the end of the line.
    for i = 1, #line do
        local character = line:sub(i, i)
        table.insert(label, alphabetMapping[character])
    end
    table.insert(labels, label)
    table.insert(transcripts, line)
end



--local function combineInputsAndTargets(inputs, targets, transcripts)
--    local inputsAndTargets = {}
--    for i = 1, #inputs do
--        table.insert(inputsAndTargets, { tensor = inputs[i], label = targets[i], truthText = transcripts[i] })
--    end
--    return inputsAndTargets
--end



local function an4Dataset(folderDirPath, audioLocationPath, targets, transcripts, windowSize, stride, nbOfSamples)
    --[[
        scan through file & make spect and combine with labels
    --]]

    local inputs = {} -- {t1, t2, ..}
    local counter = 0
    for audioPath in io.lines(audioLocationPath) do
        counter = counter + 1
        local audioData = audio.load(folderDirPath .. "/wav/" .. audioPath .. ".wav")
        -- We transpose the frequency/time to now put time on the x axis, frequency on the y axis. <-- you fool
        local spectrogram = audio.spectrogram(audioData, windowSize, 'hamming', stride):transpose(1, 2)
        table.insert(inputs, spectrogram)
        xlua.progress(counter, nbOfSamples)
    end

    local inputsAndTargets = {}
    for i = 1, #inputs do
        table.insert(inputsAndTargets, { tensor = inputs[i], label = targets[i], truthText = transcripts[i] })
    end

    return inputsAndTargets
end




-- =============================
-- Main entrance
-- =============================
function AudioData.retrieveAN4TrainingDataSet(folderDirPath, windowSize, stride)
    local audioLocationPath = folderDirPath .. "/etc/an4_train.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_train.transcription"
    local nbSamples = 948 -- [TODO make it dynamic] change Amount of samples found in the AN4 training set.

    
    -- targets={{1,2,3}, {1,2,3}, ..} each is a label for an utterance; trans={'abc bac', 'abc', ..} each is trans
    local transcripts = {}
    local targets = {}
    for line in io.lines(transcriptPath) do
        convertLineToLabels(targets, transcripts, line)
    end

    local dataSet = an4Dataset(folderDirPath, audioLocationPath, targets, transcripts, windowSize, stride, nbSamples)

    return dataSet, transcripts
end



function AudioData.retrieveAN4TestDataSet(folderDirPath, windowSize, stride, numberOfSamples)
    local audioLocationPath = folderDirPath .. "/etc/an4_test.fileids"
    local transcriptPath = folderDirPath .. "/etc/an4_test.transcription"
    local nbSamples = numberOfSamples or 130 -- Amount of samples (defaults to all).
    local targets = {}
    local transcripts = {} -- For verification of words, we return the lines of the test data.

    for line in io.lines(transcriptPath) do
        convertLineToLabels(targets, transcripts, line)
    end
    local dataSet = an4Dataset(folderDirPath, audioLocationPath, targets, transcripts, windowSize, stride, nbSamples)
    return dataSet, transcripts
end



return AudioData

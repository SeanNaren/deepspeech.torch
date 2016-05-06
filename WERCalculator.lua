require 'loader'
require 'util'
local threads = require 'threads'
local WERCalculator = {}




local function getWords(predictions, targetSentence, shouldSpellCheck, SpellingChecker, mapper)
    --[[
        Takes the resulting predictions and the transcript sentence. Returns tables of words said in both.
    --]]

    local predictionString = ""
    local prevLetter = ""
    -- Iterate through the results of the prediction and append the letter that was predicted in the sample.
    predictions = predictions:squeeze() -- Remove any single dimensions.
    for x = 1, predictions:size(1) do
        local maxValue, maxIndex = torch.max(predictions[x], 1)
        -- Minus 1 to the index because a CTC blank has the index of 0.
        maxIndex = maxIndex[1] - 1
        -- If the index is 0, that means that the character was a CTC blank.
        if (maxIndex ~= 0) then
            local letter = mapper.token2alphabet[maxIndex]
            -- We do not add the character if it is the same as the previous character.
            if (letter ~= prevLetter) then
                predictionString = predictionString .. letter
                prevLetter = letter
            end
        end
    end
    local predictedWords = {}
    for word in string.gmatch(predictionString, "%a+") do
        if (shouldSpellCheck) then
            word = SpellingChecker:correct(word)
        end
        table.insert(predictedWords, word)
    end
    local targetWords = {}
    for word in string.gmatch(targetSentence, "%a+") do
        table.insert(targetWords, word)
    end
    return predictedWords, targetWords
end



-- Calculates the word error rate (as a percentage).
function wordErrorRate(target, prediction)
    local d = torch.Tensor(#target + 1, #prediction + 1):zero()
    for i = 1, #target + 1 do
        for j = 1, #prediction + 1 do
            if (i == 1) then
                d[1][j] = j - 1
            elseif (j == 1) then
                d[i][1] = i - 1
            end
        end
    end

    for i = 2, #target + 1 do
        for j = 2, #prediction + 1 do
            if (target[i - 1] == prediction[j - 1]) then
                d[i][j] = d[i - 1][j - 1]
            else
                local substitution = d[i - 1][j - 1] + 1
                local insertion = d[i][j - 1] + 1
                local deletion = d[i - 1][j] + 1
                d[i][j] = torch.min(torch.Tensor({ substitution, insertion, deletion }))
            end
        end
    end
    return d[#target + 1][#prediction + 1] / #target * 100
end



function WERCalculator.calculateWordErrorRate(shouldSpellCheck, test_iter, SpellingChecker, model, gpu, _dir, dict_path)
    --[[
        input:
            test_iter: iter of testing
            _dir: dir where test lmdb is stored
            dict_path: path to dict
    --]]
    
    -- We collapse all the words into one large table to pass into the WER calculation.
    local totalPredictedWords = {}
    local totalTargetWords = {}

    local loader = loader(_dir)
    local indexer = indexer(_dir, 1)
    local mapper = mapper(dict_path)
    local spect_buf, label_buf, trans_buf
    local pool = threads.Threads(1,function()require 'loader'end)
    
    -- load first batch
    local inds = indexer:nxt_inds()
    pool:addjob(function()
                    return loader:nxt_batch(inds, true) -- set true to load trans
                end,
                function(spect, label, trans)
                    spect_buf=spect
                    label_buf=label
                    trans_buf=trans
                end
                )


    for i = 1, test_iter do

        pool:synchronize()
        local inputs, _, true_text = spect_buf, label_buf, trans_buf
        inds = indexer:nxt_inds()
        pool:addjob(function()
                    return loader:nxt_batch(inds, true)
                end,
                function(spect, label, trans)
                    spect_buf=spect
                    label_buf=label
                    trans_buf=trans
                end
                )

        -- We create an input of size batch x channels x freq x time (batch size in this case is 1).
        inputs = inputs:view(1, 1, inputs:size(1), inputs:size(2))

        if (gpu) then
            inputs = inputs:cuda()
        end
        local predictions = model:forward(inputs)
        local predictedWords, targetWords = getWords(predictions, true_text, shouldSpellCheck, SpellingChecker, mapper)

        for index, word in ipairs(predictedWords) do
            table.insert(totalPredictedWords, word)
        end
        for index, word in ipairs(targetWords) do
            table.insert(totalTargetWords, word)
        end
        if progress then
            xlua.progress(i, test_iter)
        end
    end

    local rate = wordErrorRate(totalTargetWords, totalPredictedWords)
    return rate
end

return WERCalculator

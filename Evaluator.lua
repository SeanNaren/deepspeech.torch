local Evaluator = {}

-- Calculates a sequence error rate (aka Levenshtein edit distance)
function Evaluator.sequenceErrorRate(target, prediction)
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
    return d[#target + 1][#prediction + 1] / #target
end

-- Turns the predictions tensor into a list of the most likely characters
function Evaluator.getPredictedCharacters(predictions)
    local predictedCharacters = {}
    local prevCharacter = 0

    -- The prediction is a sequence of likelihood vectors
    predictions = predictions:squeeze()
    local maxValues, maxIndexes = torch.max(predictions, 2)
    maxIndexes = maxIndexes:squeeze()
    for i=1,maxIndexes:size(1) do
        -- If the index is 1, that means that the prediction was a blank label
        local character = maxIndexes[i] - 1 -- Caveat about CTC indexes and our labeling scheme
        if (character ~= 0) then
            -- We do not add the phone if it is the same as the previous phone.
            if (character ~= prevCharacter) then
                table.insert(predictedCharacters, character)
                prevCharacter = character
            end
        end
    end

    return predictedCharacters
end


return Evaluator
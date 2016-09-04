local SequenceError = torch.class("SequenceError")

-- Calculates a sequence error rate (aka Levenshtein edit distance)
function SequenceError:sequenceErrorRate(target, prediction)
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
    local errorRate = d[#target + 1][#prediction + 1] / #target
    return errorRate
end

function SequenceError:calculateCER(targetTranscript, predictTranscript)
    return self:sequenceErrorRate(targetTranscript, predictTranscript)
end

function SequenceError:calculateWER(targetTranscript, predictTranscript)
    -- convert to words before calculation
    local targetWords = {}
    for word in targetTranscript:gmatch("%S+") do table.insert(targetWords, word) end
    local predictedWords = {}
    for word in predictTranscript:gmatch("%S+") do table.insert(predictedWords, word) end
    return self:sequenceErrorRate(targetWords, predictedWords)
end

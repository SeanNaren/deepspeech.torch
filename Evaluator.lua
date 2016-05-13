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


function Evaluator.predict2tokens(predictions, mapper)
    --[[
        Turns the predictions tensor into a list of the most likely tokens

        NOTE:
            to compute WER we strip the begining and ending spaces
    --]]
    local tokens = {}
    local blank_token = mapper.alphabet2token['$']
    local space_token = mapper.alphabet2token[' ']
    local strip_head = true
    local pre_token = blank_token


    -- The prediction is a sequence of likelihood vectors
    local _, maxIndexes = torch.max(predictions, 2)
    maxIndexes = maxIndexes:squeeze()

    for i=1,maxIndexes:size(1) do
        local token = maxIndexes[i] - 1 -- CTC indexes start from 1, while token starts from 0
        -- add token if it's not blank, and is not the same as pre_token
        if token ~= blank_token and token ~= pre_token then
            if token ~= space_token then strip_head = false end
            if not strip_head then table.insert(tokens, token) end
            pre_token = token
        end
    end
    
    if tokens[#tokens] == space_token then table.remove(tokens, #tokens) end

    return tokens
end


return Evaluator

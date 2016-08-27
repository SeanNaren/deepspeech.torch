require 'torch'

-- construct an object to deal with the mapping
local mapper = torch.class('Mapper')

function mapper:__init(dictPath)
    assert(paths.filep(dictPath), dictPath ..' not found')

    self.alphabet2token = {}
    self.token2alphabet = {}

    -- make maps
    local cnt = 0
    for line in io.lines(dictPath) do
        self.alphabet2token[line] = cnt
        self.token2alphabet[cnt] = line
        cnt = cnt + 1
    end
end

function mapper:encodeString(line)
    local label = {}
    for i = 1, #line do
        local character = line:sub(i, i)
        table.insert(label, self.alphabet2token[character])
    end
    return label
end

function mapper:decodeOutput(predictions)
    --[[
        Turns the predictions tensor into a list of the most likely tokens

        NOTE:
            to compute WER we strip the begining and ending spaces
    --]]
    local tokens = {}
    local blankToken = self.alphabet2token['$']
    local preToken = blankToken
    -- The prediction is a sequence of likelihood vectors
    local _, maxIndices = torch.max(predictions, 2)
    maxIndices = maxIndices:float():squeeze()

    for i=1, maxIndices:size(1) do
        local token = maxIndices[i] - 1 -- CTC indexes start from 1, while token starts from 0
        -- add token if it's not blank, and is not the same as pre_token
        if token ~= blankToken and token ~= preToken then
            table.insert(tokens, token)
        end
        preToken = token
    end
    return tokens
end

function mapper:tokensToText(tokens)
    local text = ""
    for i, t in ipairs(tokens) do
        text = text .. self.token2alphabet[tokens[i]]
    end
    return text
end

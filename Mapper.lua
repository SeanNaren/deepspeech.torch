require 'torch'
require 'string'

-- construct an object to deal with the mapping
local mapper = torch.class('Mapper')

function mapper:__init(dictPath)
    assert(paths.filep(dictPath), dictPath ..' not found')

    self.alphabet2token = {}
    self.token2alphabet = {}

    -- make maps
    local cnt = 0
    for line in io.lines(dictPath) do
        line = string.lower(line)
        self.alphabet2token[line] = cnt
        self.token2alphabet[cnt] = line
        cnt = cnt + 1
    end
end

require 'torch'

-- construct an object to deal with the mapping
local mapper = torch.class('Mapper')

function mapper:__init(dict_path)
    assert(paths.filep(dict_path), dict_path..' not found')

    self.alphabet2token = {}
    self.token2alphabet = {}

    -- make maps
    local cnt = 0
    for line in io.lines(dict_path) do
        self.alphabet2token[line] = cnt
        self.token2alphabet[cnt] = line
        cnt = cnt + 1
    end
end

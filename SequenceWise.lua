------------------------------------------------------------------------
--[[ SequenceWise ]] --
-- Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
-- Allows handling of variable sequence lengths and minibatch sizes.
------------------------------------------------------------------------

local SequenceWise, parent = torch.class('nn.SequenceWise', 'nn.Sequential')

function SequenceWise:__init(module)
    parent.__init(self)

    self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
    self.view_out = nn.View(1, -1):setNumInputDims(2)

    self:add(self.view_in)
    self:add(module)
    self:add(self.view_out)
end

function SequenceWise:updateOutput(input)
    local T, N = input:size(1), input:size(2)
    self.view_in:resetSize(T * N, -1)
    self.view_out:resetSize(T, N, -1)
    return parent.updateOutput(self, input)
end


function SequenceWise:__tostring__()
    local tab = '  '
    local line = '\n'
    local next = ' -> '
    local str = 'nn.SequenceWise'
    str = str .. ' {' .. line .. tab .. '[input'
    for i=1,#self.modules do
        str = str .. next .. '(' .. i .. ')'
    end
    str = str .. next .. 'output]'
    for i=1,#self.modules do
        str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
    end
    str = str .. line .. '}'
    return str
end
--[[
-- Reverses a sequence on a given dimension.
-- Example: Given a tensor of torch.Tensor({{1,2,3,4,5}, {6,7,8,9,10})
-- nn.ReverseSequence(1):forward(tensor) would give: torch.Tensor({{6,7,8,9,10},{1,2,3,4,5}})
 ]]
local ReverseSequence, parent = torch.class("nn.ReverseSequence", "nn.Module")

function ReverseSequence:__init(dim)
    parent.__init(self)
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    assert(dim, "Must specify dimension to reverse sequence over")
    assert(dim <= 3, "Dimension has to be no greater than 3 (Only supports up to 3d).")
    self.dim = dim
end

function ReverseSequence:reverseOutput(input, output)
    output:resize(input:size()):zero()
    -- reverse output
    local k = 1
    for i = input:size(1), 1, -1 do
        output[k] = input[i]
        k = k + 1
    end
end


function ReverseSequence:updateOutput(input)
    if (self.dim == 1) then
        ReverseSequence:reverseOutput(input, self.output)
    end
    if (self.dim == 2) then
        input = input:transpose(1, 2)
        ReverseSequence:reverseOutput(input, self.output)
        self.output = self.output:transpose(1, 2)
    end
    if (self.dim == 3) then
        input = input:transpose(1, 3)
        ReverseSequence:reverseOutput(input, self.output)
        self.output = self.output:transpose(1, 3)
    end
    return self.output
end

function ReverseSequence:reverseGradOutput(gradOutput, gradInput)
    gradInput:resize(gradOutput:size()):zero()
    local k = 1
    for i = gradOutput:size(1), 1, -1 do
        gradInput[k] = gradOutput[i]
        k = k + 1
    end
end

function ReverseSequence:updateGradInput(inputTable, gradOutput)
    if (self.dim == 1) then
        ReverseSequence:reverseGradOutput(gradOutput, self.gradInput)
    end
    if (self.dim == 2) then
        gradOutput = gradOutput:transpose(1, 2)
        ReverseSequence:reverseGradOutput(gradOutput, self.gradInput)
        self.gradInput = self.gradInput:transpose(1, 2)
    end
    if (self.dim == 3) then
        gradOutput = gradOutput:transpose(1, 3)
        ReverseSequence:reverseGradOutput(gradOutput, self.gradInput)
        self.gradInput = self.gradInput:transpose(1, 3)
    end
    return self.gradInput
end
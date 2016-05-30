require 'nn'

local test = torch.TestSuite()
local mytester

function test.MaskRNN()
    require '../MaskRNN'
    local N = 2
    local T = 3
    local H = 4
    local seqLength = torch.Tensor({ 1, 2 }) -- real sequence lengths of each batch.
    local identity = nn.Identity()
    local mask = nn.MaskRNN(identity)
    local input = torch.rand(T, N, H)
    local output = mask:forward({ input, seqLength })
    output = output:view(input:size(1), input:size(2), input:size(3))
    local expectedOutput = input:clone()
    expectedOutput[2][1]:fill(0)
    expectedOutput[3]:fill(0)
    local gradInput = torch.rand(T, N, H)
    local grads = mask:backward({ input, seqLength }, gradInput)

    local expectedGrads = gradInput:clone()
    expectedGrads[2][1]:fill(0)
    expectedGrads[3]:fill(0)
    mytester:eq(output, expectedOutput, 'masking of outputs was incorrect')
    mytester:eq(grads, expectedGrads, 'masking of gradients was incorrect')
end

function test.reverseMaskRNN()
    require '../ReverseMaskRNN'
    local N = 2
    local T = 3
    local H = 4
    local seqLength = torch.Tensor({ 1, 2 }) -- real sequence lengths of each batch.
    local identity = nn.Identity()
    local mask = nn.ReverseMaskRNN(identity)
    local input = torch.rand(T, N, H)
    local output = mask:forward({ input, seqLength })
    output = output:view(input:size(1), input:size(2), input:size(3))
    local expectedOutput = input:clone()
    expectedOutput[2][1]:fill(0)
    expectedOutput[3]:fill(0)
    local gradInput = torch.rand(T, N, H)
    local grads = mask:backward({ input, seqLength }, gradInput)

    local expectedGrads = gradInput:clone()
    expectedGrads[2][1]:fill(0)
    expectedGrads[3]:fill(0)
    mytester:eq(output, expectedOutput, 'masking of outputs was incorrect')
    mytester:eq(grads, expectedGrads, 'masking of gradients was incorrect')

    -- Test reverse sequence method
    input = torch.rand(4, 1, 1)
    seqLength = torch.Tensor({ 3 })
    local output = mask:reverse(input, seqLength)
    local expectedOutput = input:clone()
    for i = 1, seqLength[1] do
        expectedOutput[i] = input[seqLength[1] - (i - 1)]
    end
    expectedOutput[4]:fill(0)
    mytester:eq(output, expectedOutput, 'reversing of input was incorrect')
end

mytester = torch.Tester()
mytester:add(test)
mytester:run()
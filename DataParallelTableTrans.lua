---- inherit DataParallel table
local DataParallelTableTrans = torch.class('nn.DataParallelTableTrans','nn.DataParallelTable')

local function hasFlattenedParmeters(self)
   if not self.flattenedParams then
      return false
   end
   for _, param in ipairs(self.modules[1]:parameters()) do
      if param:storage() ~= self.flattenedParams[1][1]:storage() then
         return false
      end
   end
   return true
end

-- extracts the value at idx from each entry in tbl
local function pluck(tbl, idx)
   local r = {}
   for n, val in ipairs(tbl) do
      r[n] = val[idx]
   end
   return r
end

function DataParallelTableTrans:updateOutput(input)
   if self.flattenParams and not hasFlattenedParmeters(self) then
      self:flattenParameters()
   end
   if self.needsSync then
      self:syncParameters()
   end

   local prevGpuid = cutorch.getDevice()

   -- distribute the input to GPUs
   self:_distribute(self.inputGpu, input)

   -- update output for each module
   local inputGpu = self.inputGpu
   self.outputGpu = self.impl:exec(function(m, i)
      if torch.isTensor(inputGpu[i]) and inputGpu[i]:numel() == 0 then
         return torch.CudaTensor()
      else
         return m:updateOutput(inputGpu[i])
      end
   end)

   self.output = self.outputGpu
   cutorch.setDevice(prevGpuid)

   return self.output
end

function DataParallelTableTrans:backward(input, target, scale)
   return self:__backward_inner('backward', input, target, scale)
end

function DataParallelTableTrans:updateGradInput(input, target)
   return self:__backward_inner('updateGradInput', input, target)
end

local function slice(tbl, first, last, step)
    local sliced
    if torch.type(tbl) == 'table' then
        sliced = {}
        for i = first or 1, last or #tbl, step or 1 do
            sliced[#sliced+1] = tbl[i]
        end
    else
        sliced = torch.Tensor()
        sliced:resize(last-first+1):copy(sizes:select(first,last))
    end
    return sliced
end

function DataParallelTableTrans:__backward_inner(method, input, target, scale)
   local prevGpuid = cutorch.getDevice()
   local inputGpu = self.inputGpu
   local outputGpu = self.outputGpu
   local targetGpu, sizeGpu = {}, {}

   -- distribute the target to GPUs
   for i = 1, #self.gpuAssignments do
     table.insert(sizeGpu, inputGpu[i][2])
   end
   local batch_size = inputGpu[1][1]:size(1)
   local loss = torch.Tensor(#self.gpuAssignments)
   self.gradInputGpu = self.impl:exec(function(m, i)
      if torch.isTensor(inputGpu[i]) and inputGpu[i]:numel() == 0 then
         return torch.CudaTensor()
      else
         require 'nnx'
         local ctcCriterion = nn.CTCCriterion():cuda()
         local targets_slice = slice(target, 1+(i-1)*batch_size, i*batch_size)
         loss[i] = ctcCriterion:forward(outputGpu[i], targets_slice, sizeGpu[i])
         local gradOutput = ctcCriterion:backward(outputGpu[i], targets_slice)
         return m[method](m, inputGpu[i], gradOutput, scale)
      end
   end)
   if method == 'backward' then
      local params = self:moduleParameters()
      -- Accumulate the gradients onto the base GPU
      if self.flattenedParams and self.usenccl and not cudaLaunchBlocking then
         if #self.gpuAssignments > 1 then
            nccl.reduce(pluck(self.flattenedParams, 2), nil, true, 1)
         end
      else
         self:_reduce(pluck(params, 2))
      end
      -- Zero out gradients on the other GPUs
      for i = 2, #self.gpuAssignments do
         cutorch.setDevice(self.gpuAssignments[i])
         for _, gradParam in ipairs(params[i][2]) do
            gradParam:zero()
         end
      end
      self.needsSync = true
   end
   cutorch.setDevice(prevGpuid)
   return loss:mean()
end
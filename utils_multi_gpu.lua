require 'cunn'
require 'rnn'
require 'nngraph'
require 'MaskRNN'
require 'ReverseRNN'
require 'cudnn'
local ffi=require 'ffi'

---- inherit DataParallel table
local DataParallelTableTrans = torch.class('nn.DataParallelTableTrans','nn.DataParallelTable')

function DataParallelTableTrans:__init(dimen)
    nn.DataParallelTable.__init(self, dimen) -- call the parent init
end


local function sumSizes(tensors, dim)
    local size
    for i=1,#tensors do
        if tensors[i]:numel() > 0 then
            if size then
                size[dim] = size[dim] + tensors[i]:size(dim)
            else
                size = tensors[i]:size()
            end
        end
    end
    return size
end

-- reload _concat to resolve the bug
function DataParallelTableTrans:_concatTensorRecursive(dst, src)

    return src

--    if torch.type(src[1]) == 'table' then
--        if torch.type(dst) ~= 'table' or #src[1] ~= #dst then
--            dst = {}
--        end
--        for i, _ in ipairs(src[1]) do
--            dst[i] = self:_concatTensorRecursive(dst[i], pluck(src, i))
--        end
--        return dst
--    end
--    
--    assert(torch.isTensor(src[1]), 'input must be a tensor or table of tensors')
----    for i= 1,#src do
----        print(i..'th mat size')
----        print(src[i]:size())
----        src[i] = src[i]:view(-1,5,28)
----        src[i] = src[i]:transpose(1,2):contiguous()
----        src[i] = src[i]:view(-1,28)
----        print(src[i]:size())
----    end
--    
--    cutorch.setDevice(self.gpuAssignments[1])
--    dst = torch.type(dst) == 'torch.CudaTensor' and dst or torch.CudaTensor()
--
--    local cumsum = sumSizes(src, self.dimension)
--
--    if cumsum == nil then return dst end
--
--    dst:resize(cumsum)
--
--    local start = 1
--    for i, s in ipairs(src) do
--        if torch.numel(s) > 0 then
--            local sz = s:size(self.dimension)
--            dst:narrow(self.dimension, start, sz):copy(s)
--            start = start + sz
--        end
--    end
-- --   print('dst size is')
----    dst = dst:view(20 ,-1, 28)
----    dst = dst:transpose(1,2):contiguous()
----    dst = dst:view(-1,28)
----    print(dst:size())
--    return dst
end



function DataParallelTableTrans:__backward(method, input, gradOutput, scale)
   local prevGpuid = cutorch.getDevice()
   local inputGpu, gradOutputGpu = self.inputGpu, self.gradOutputGpu

   if method == 'backward' or method == 'updateGradInput' then
      -- distribute the gradOutput to GPUs
      -- TODO self:_distribute(self.gradOutputGpu, gradOutput)
      self.gradOutputGpu = gradOutput

      self.gradInputGpu = self.impl:exec(function(m, i)
         if torch.isTensor(inputGpu[i]) and inputGpu[i]:numel() == 0 then
            return torch.CudaTensor()
         else
            return m[method](m, inputGpu[i], gradOutputGpu[i], scale)
         end
      end)

      if self.gradInput then
         -- concatenate the gradInput to the base GPU
         self.gradInput = self:_concat(self.gradInput, self.gradInputGpu)
      end
   end

   if method == 'accGradParameters' then
      self.impl:exec(function(m, i)
         if torch.isTensor(inputGpu[i]) and inputGpu[i]:numel() == 0 then
            return torch.CudaTensor()
         else
            return m:accGradParameters(inputGpu[i], gradOutputGpu[i], scale)
         end
      end)
   end

   if method == 'backward' or method == 'accGradParameters' then
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
    return self.gradInput
end




local default_GPU = 1
function makeDataParallel(model, nGPU, is_cudnn)
   -- if nGPU <= 0 then return model end
   -- if nGPU > 1 then
   --    print('converting module to nn.DataParallelTable')
   --    assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
   --    local model_single = model
   --    model = nn.DataParallelTable(1)
   --    for i=1, nGPU do
   --       cutorch.setDevice(i)
   --       model:add(model_single:clone():cuda(), i)
   --    end
   -- end
   -- cutorch.setDevice(default_GPU)
   
    if nGPU >= 1 then
        
        if is_cudnn then
            cudnn.fastest = true
            model = cudnn.convert(model, cudnn)
        end

        gpus = torch.range(1, nGPU):totable()

        dpt = nn.DataParallelTableTrans(1)
        dpt:add(model, gpus) -- now use our impl instead; nn.DataParallelTable(1)                  
        dpt:threads(function()
                     require 'nngraph'
                     require 'MaskRNN'
                     require 'ReverseRNN'
                     if is_cudnn then
                        local cudnn = require 'cudnn'
                        cudnn.fastest = true
                        -- cudnn.benchmark = true
                     else
                        require 'rnn'
                     end
                  end)
        dpt.gradInput = nil
        model = dpt
        model:cuda()
    end

    return model

end

local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTableTrans(1)
   cutorch.setDevice(default_GPU)
   newDPT:add(module:get(1), default_GPU)
   return newDPT
end

function saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, cleanDPT(model))
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(cleanDPT(module))
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model)
   elseif torch.type(model) == 'nn.gModule' then
      torch.save(filename, model)
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function loadDataParallel(filename, nGPU, is_cudnn)
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU, is_cudnn)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU, is_cudnn)
         end
      end
      return model
   elseif torch.type(model) == 'nn.gModule' then
      model = makeDataParallel(model, nGPU, is_cudnn)
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

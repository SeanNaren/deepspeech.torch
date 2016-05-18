require 'cunn'
-- require 'rnn'
require 'nngraph'
-- require 'MaskRNN'
-- require 'ReverseRNN'
require 'cudnn'
local ffi=require 'ffi'

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
      if nGPU > 1 then
         gpus = torch.range(1, nGPU):totable()
         dpt = nn.DataParallelTable(1)
                 :add(model, gpus)
             :threads(function()
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
      end
      model:cuda()
   end
   return model
end

local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1)
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
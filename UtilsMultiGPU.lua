require 'cunn'
require 'rnn'
require 'nngraph'
require 'MaskRNN'
require 'ReverseMaskRNN'
require 'cudnn'
require 'DataParallelTableTrans'
local default_GPU = 1
function makeDataParallel(model, nGPU, is_cudnn)
    if nGPU >= 1 then
        if is_cudnn then
            cudnn.fastest = true
            model = cudnn.convert(model, cudnn)
        end
        if nGPU > 1 then
            gpus = torch.range(1, nGPU):totable()

            dpt = nn.DataParallelTableTrans(1, true, true)
            dpt:add(model, gpus) -- now use our impl instead; nn.DataParallelTable(1)
            dpt:threads(function()
                         require 'nngraph'
                         require 'MaskRNN'
                         require 'ReverseMaskRNN'
                         require 'rnn'
                         require 'cudnn'
                      end)
            dpt.gradInput = nil
            model = dpt
        end
        model:cuda()
    end
    return model
end

function saveDataParallel(filename, model)
   local model_type = torch.type(model)
   model:clearState()
   if model_type == 'nn.DataParallelTable' or
      model_type == 'nn.DataParallelTableTrans' then
      model = model:get(1)
   elseif model_type == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' or
            torch.type(module) == 'nn.DataParallelTableTrans' then
            temp_model:add(module:get(1))
         else
            temp_model:add(module)
         end
      end
      model = temp_model
   else
      assert(model_type == 'nn.gModule',
        'This saving function only works with Sequential, gModule or DataParallelTable modules.')
   end
   if torch.type(model) == 'nn.gModule' then
      for _,node in ipairs(model.backwardnodes) do
         if node.data.module then
            node.data.module.reverse_gradOutput = nil
            node.data.module._gradOutput = nil
         end
      end
      for _,node in ipairs(model.forwardnodes) do
         if node.data.module then
            node.data.module.reverse_input = nil
            node.data.module._input = nil
         end
      end
   end
   torch.save(filename, model)
end

function loadDataParallel(filename, nGPU, is_cudnn)
   local model = torch.load(filename)
   local model_type = torch.type(model)
   if model_type == 'nn.DataParallelTable' or
      model_type == 'nn.DataParallelTableTrans' then
      return makeDataParallel(model:get(1):float(), nGPU, is_cudnn)
   elseif model_type == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' or
            torch.type(module) == 'nn.DataParallelTableTrans' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU, is_cudnn)
         end
      end
      return model
   elseif model_type == 'nn.gModule' then
      model = makeDataParallel(model, nGPU, is_cudnn)
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

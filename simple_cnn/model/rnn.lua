require 'torch'
require 'nn'
require 'nngraph'
require 'rnn'


-- -- hyper-parameters 
-- batchSize = 8
-- nIndex = 100 -- input words
-- nClass = 7 -- output classes
-- lr = 0.1


local ModelBuilder = torch.class('ModelBuilder')

function ModelBuilder:make_net(w2v)

	if opt.cudnn == 1 then
		require 'cudnn'
		require 'cunn'
	end

	local arg1 = nn.Identity()()
	local arg2 = nn.Identity()()

	local hiddenSize = opt.config.h or opt.vec_size
	local rho = opt.config.rho or opt.max_sent
	local nonLinear = nn.Sigmoid()

	local lookup = nn.LookupTable(opt.vocab_size, opt.vec_size)

	if opt.w2v ~= '' then
		lookup.weight:copy(w2v)
	end

	-- padding should always be 0
	lookup.weight[1]:zero()

	local W_hh = nn.Linear(hiddenSize, hiddenSize)
	W_hh.weight = torch.eye(hiddenSize)

	-- build simple recurrent neural network
	-- rec(sizeOfOutput, input, W_(hh), nonLinear, maxPropStep)
	local r = nn.Recurrent(
		hiddenSize, nn.Identity(), 
		W_hh, nonLinear, 
		rho
	)

	local linear = nn.Linear(hiddenSize, opt.num_classes)
	local w = math.sqrt(6/(hiddenSize+opt.num_classes))

	linear.weight:uniform(-w, w)
	linear.bias:zero()

	local softmax
	if opt.cudnn == 1 then
		softmax = cudnn.LogSoftMax()
	else
		softmax = nn.LogSoftMax()
	end

	local rnn = nn.Sequential()
		:add(nn.JoinTable(1,1))
		:add(lookup)
		:add(nn.SplitTable(1,2))
		:add(nn.Sequencer(r))
		:add(nn.SelectTable(-1)) -- this selects the last time-step of the rnn output sequence
		:add(nn.Dropout(opt.dropout_p))
		:add(linear)
		:add(softmax)

	local output = rnn({arg1, arg2})

	-- model = nn.gModule({arg1}, {output})
	model = nn.gModule({arg1, arg2}, {output})

	return model

end

return ModelBuilder


-- A from-scratch model (template), not yet implemented

-- local RNN = {}

-- function RNN.rnn(input_size, rnn_size, n, dropout)
	
-- 	-- there are n+1 inputs (hiddens on each layer and x)
-- 	local inputs = {}
-- 	table.insert(inputs, nn.Identity()()) -- x
-- 	for L = 1,n do
-- 		table.insert(inputs, nn.Identity()()) -- prev_h[L]

-- 	end

-- 	local x, input_size_L
-- 	local outputs = {}
-- 	for L = 1,n do
		
-- 		local prev_h = inputs[L+1]
-- 		if L == 1 then 
-- 			x = OneHot(input_size)(inputs[1])
-- 			input_size_L = input_size
-- 		else 
-- 			x = outputs[(L-1)] 
-- 			if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
-- 			input_size_L = rnn_size
-- 		end

-- 		-- RNN tick
-- 		local i2h = nn.Linear(input_size_L, rnn_size)(x)
-- 		local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
-- 		local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

-- 		table.insert(outputs, next_h)
-- 	end
-- -- set up the decoder
-- 	local top_h = outputs[#outputs]
-- 	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
-- 	local proj = nn.Linear(rnn_size, input_size)(top_h)
-- 	local logsoft = nn.LogSoftMax()(proj)
-- 	table.insert(outputs, logsoft)

-- 	return nn.gModule(inputs, outputs)
-- end

-- return RNN
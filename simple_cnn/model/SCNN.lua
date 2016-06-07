require 'torch'
require 'nn'
require 'nngraph'



-- nngraph.setDebug(true)


local ModelBuilder = torch.class('ModelBuilder')

function ModelBuilder:make_net(w2v)

	if opt.cudnn == 1 then
		require 'cudnn'
		require 'cunn'
	end
	local arg1 = nn.Identity()()
	local arg2 = nn.Identity()()

	local lookup
	lookup = nn.LookupTable(opt.vocab_size, opt.vec_size)
	
	if opt.w2v ~= '' then
		lookup.weight:copy(w2v)
	end

	-- padding should always be 0
	lookup.weight[1]:zero()
	

	local conv1, conv2

	conv1 = nn.ConcatTable()
	conv1:add(nn.Mean(2))
	conv1:add(nn.Max(2))
	conv1:add(nn.Min(2))
	conv2 = nn.ConcatTable()
	conv2:add(nn.Mean(2))
	conv2:add(nn.Max(2))
	conv2:add(nn.Min(2))


	local c1 = nn.Sequential()
	local c2 = nn.Sequential()

	c1:add(lookup)
	c1:add(conv1)
	c1:add(nn.JoinTable(2))

	c2:add(lookup)
	c2:add(conv2)
	c2:add(nn.JoinTable(2))

	local conv_ = nn.ParallelTable()

	conv_:add(c1)
	conv_:add(c2)

	local conv = nn.Sequential()

	conv:add(conv_)
	conv:add(nn.JoinTable(2))

	local last_layer
	if opt.cudnn == 1 then
		last_layer = nn.Normalize(2)(cudnn.Tanh()(conv({arg1, arg2})))
	else
		last_layer = nn.Normalize(2)(nn.Tanh()(conv({arg1, arg2})))
	end

	-- simple MLP layer
	local linear = nn.Linear(opt.vec_size*6, opt.num_classes)
	-- linear.weight:normal():mul(0.01)
	local r = math.sqrt(6/(opt.vec_size*6+opt.num_classes))

	linear.weight:uniform(-r, r)
	linear.bias:zero()
	-- print(linear.weight:size())

	local softmax
	if opt.cudnn == 1 then
		softmax = cudnn.LogSoftMax()
	else
		softmax = nn.LogSoftMax()
	end

	local output = softmax(linear(nn.Dropout(opt.dropout_p)(last_layer))) 

	-- model = nn.gModule({arg1}, {output})
	model = nn.gModule({arg1, arg2}, {output})

	return model
end


return ModelBuilder
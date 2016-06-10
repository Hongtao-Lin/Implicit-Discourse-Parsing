require 'torch'
require 'nn'
require 'nngraph'

local ModelBuilder = torch.class('ModelBuilder')

function ModelBuilder:make_net(w2v)

	if opt.cudnn == 1 then
		require 'cudnn'
		require 'cunn'
	end

	local arg1 = nn.Identity()()
	local arg2 = nn.Identity()()

	local pos1, pos2, pos_input, pos_lookup


	local input = nn.JoinTable(1, 1)({arg1, arg2})

	local lookup
	if opt.POS_concat then
		lookup = nn.LookupTable(opt.vocab_size, opt.vec_size-opt.POS_dim)
	else
		lookup = nn.LookupTable(opt.vocab_size, opt.vec_size)
	end
	lookup.weight:copy(w2v)
	lookup.weight[1]:zero()

	-- local a = nn.Reshape(opt.max_sent, opt.vec_size, true)(lookup(arg1))
	-- local b = nn.Reshape(opt.max_sent, opt.vec_size, true)(lookup(arg2))
	-- local input = nn.JoinTable(1, 2)({a, b})
	-- local input = nn.JoinTable(1, 2)({lookup(arg1), lookup(arg2)})

	input = lookup(input)

	if opt.POS_concat then
		pos_lookup = nn.LookupTable(opt.POS_dim+1, opt.POS_dim)
		local pos_weight = torch.zeros(opt.POS_dim+1, opt.POS_dim)
		pos_weight[{{2,-1},{}}] = torch.eye(opt.POS_dim, opt.POS_dim)
		pos_lookup.weight = pos_weight
		pos1 = nn.Identity()()
		pos2 = nn.Identity()()
		local p1 = nn.Reshape(opt.max_sent, opt.POS_dim, true)(pos_lookup(pos1))
		pos_lookup.name = 'pos_w2v'
		pos_input = nn.JoinTable(1, 2)({p1, pos_lookup(pos2)})
		-- pos_input = nn.Reshape(opt.max_sent, opt.POS_dim, true)(pos_input)
		input = nn.JoinTable(2, 2)({input, pos_input})
	end

	local conv
	local kernels = opt.config.K or {6}
	local conv_frame = opt.config.F or 500
	local conv_layers = {}

	local conv_per_frame = math.floor(conv_frame / #kernels)
	conv_frame = conv_per_frame * #kernels

	for i = 1, #kernels do
		if opt.cudnn == 1 then
			conv = cudnn.TemporalConvolution(opt.vec_size, conv_per_frame, kernels[i])
		else
			conv = nn.TemporalConvolution(opt.vec_size, conv_per_frame, kernel[i])
		end
		conv.weight:normal(-0.01, 0.01)
		conv.bias:zero()
		
		conv_layer = conv(input)
		if opt.cudnn == 1 then
			max_time = nn.Max(2)(cudnn.Tanh()(conv_layer)) -- max over time
		else
			max_time = nn.Max(2)(nn.Tanh()(conv_layer)) -- max over time
		end
		
		table.insert(conv_layers, max_time)	
	end

	local last_layer

	if #conv_layers == 1 then
		last_layer = conv_layers[1]
	else
		last_layer = nn.JoinTable(2)(conv_layers)
	end

	local linear = nn.Linear(conv_frame, opt.num_classes)

	-- simple MLP layer
	local r = math.sqrt(6/(conv_frame+opt.num_classes))

	linear.weight:uniform(-r, r)
	linear.bias:zero()

	local softmax
	if opt.cudnn == 1 then
		softmax = cudnn.LogSoftMax()
	else
		softmax = nn.LogSoftMax()
	end

	local output = softmax(linear(nn.Dropout(opt.dropout_p)(last_layer)))
	if opt.POS_concat then
		model = nn.gModule({arg1, arg2, pos1, pos2}, {output})
	else
		model = nn.gModule({arg1, arg2}, {output})
	end
	return model
end

return ModelBuilder

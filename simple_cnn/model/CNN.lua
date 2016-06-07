require 'torch'
require 'nn'
require 'nngraph'

nngraph.setDebug(true)

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

	local conv
	local concat_in

	local a = nn.Reshape(opt.max_sent, opt.vec_size, true)(lookup(arg1))
	local b = nn.Reshape(opt.max_sent, opt.vec_size, true)(lookup(arg2))

	local j = nn.JoinTable(2, 2)({a, b})

	concat_in = nn.Reshape(1, opt.max_sent, opt.vec_size*2, true)(j)

	local conv_w = opt.config.W or 3
	local conv_h = opt.config.H or 3
	local fold_num = opt.config.F or 1

	if opt.cudnn == 1 then
		conv = cudnn.SpatialConvolution(1, 1, conv_w, conv_h)
	else
		conv = nn.SpatialConvolution(1, 1, conv_w, conv_h)
	end

	conv.weight = torch.randn(conv_w, conv_h)
	conv.bias:zero()

	local conved_w = opt.vec_size*2-conv_w+1
	local conved_h = opt.max_sent-conv_h+1

	conv = nn.Reshape(conved_h, conved_w, true)(conv(concat_in))


	local max_layer = nn.TemporalMaxPooling(conved_h-fold_num+1, 1)(conv)

	local fold_layer = nn.CAddTable()(nn.SplitTable(2)(max_layer))

	local non_linear
	if opt.cudnn == 1 then
		non_linear = (cudnn.Tanh()(fold_layer))
	else
		non_linear = (nn.Tanh()(fold_layer))
	end

	local linear = nn.Linear(conved_w, opt.num_classes)

	-- simple MLP layer
	local r = math.sqrt(6/(conved_w+opt.num_classes))

	linear.weight:uniform(-r, r)
	linear.bias:zero()

	local softmax
	if opt.cudnn == 1 then
		softmax = cudnn.LogSoftMax()
	else
		softmax = nn.LogSoftMax()
	end

	local output = softmax(linear(nn.Dropout(opt.dropout_p)(non_linear))) 
	model = nn.gModule({arg1, arg2}, {output})
	return model
end

return ModelBuilder

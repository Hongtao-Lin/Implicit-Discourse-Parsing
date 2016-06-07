require 'hdf5'
require 'nn'
require 'torch'
require 'optim'
require 'lfs'
require "nngraph"


cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Convolutional net for sentence classification')
cmd:text()
cmd:text('Options')
cmd:option('-train', '', 'Training data')
cmd:option('-dev', '', 'Dev data')
cmd:option('-test', '', 'Dev data')
cmd:option('-model', '', 'Warm start model if test')
cmd:option('-savefile', 'results/model.t7', 'path for save')
cmd:option('-w2v', 'data/w2v.hdf5', 'Word2vec data')
cmd:option('-mode', '', 'Whether update w2v parameters.')
cmd:option('-w2i', 'data/vocab.save', 'Word2idx data')
cmd:option('-dropout_p', 0.5, 'p for dropout')
cmd:option('-L2s', 2, 'Normalization for the final layer')
cmd:option('-mtype', 'SCNN', 'model type')
cmd:option('-config', '{}', 'Config per model')
cmd:option('-num_classes', 11, 'Number of classes')
cmd:option('-vec_size', 50, 'Vec dimension')
cmd:option('-vocab_size', 21244, 'Vocab dimension')
cmd:option('-max_sent', 150, 'Max sentence length')
cmd:option('-train_only', 1, 'Tranining or testing')
cmd:option('-batch_size', 100, 'Number of bathces')
cmd:option('-epochs', 100, 'num of epoch')
cmd:option('-optim', 'adagrad', 'method for optimizarion')
cmd:option('-cudnn', 0, 'use cuda or not')
cmd:option('-debug', 1, 'debug mode')

function get_layer(model, name)
	local named_layer
	function get(layer)
		if torch.typename(layer) == name or layer.name == name then
			named_layer = layer
		end
	end

	model:apply(get)
	return named_layer
end

function build_model(w2v)
	local ModelBuilder

	if opt.cudnn == 1 then
		require "cudnn"
		require "cunn"
	end

	ModelBuilder = require ('model.' .. opt.mtype)
	local model_builder = ModelBuilder.new()

	local model
	if opt.model == '' then
		model = model_builder:make_net(w2v)
	else
		print(opt.model)
		model = torch.load(opt.model).model
	end

	-- local w = torch.Tensor({0.08, 0.02, 0.01, 0.03, 0.05, 0.2, 0.02, 0.25, 0.25, 0.01, 0.15})
	-- local w = torch.Tensor({0.20, 0.02, 0.01, 0.03, 0.05, 0.2, 0.02, 0.20, 0.20, 0.01, 0.20})

	local criterion = nn.ClassNLLCriterion(w)
	
	local layers = {}
	layers['linear'] = get_layer(model, 'nn.Linear')
	layers['w2v'] = get_layer(model, 'nn.LookupTable')
	if opt.mtype == "RNN" then
		layers["W_hh"] = get_layer(model, "nn.Recurrent")
	end
	-- move to GPU
	if opt.cudnn == 1 then
		model = model:cuda()
		criterion = criterion:cuda()
	end

	return model, criterion, layers
end

function load_data()
	local train, train_label
	local dev, dev_label
	local test, test_label
	local arg1, arg2

	local w2v
	local word_to_idx = {}
	print('loading data...')
	local f = hdf5.open(opt.w2v, 'r')
	w2v = f:read('w2v'):all()
	-- f = io.open(opt.w2i, 'r')
	local n = 1
	for line in io.lines(opt.w2i) do
		word_to_idx[line] = n
		n = n + 1
	end

	if opt.train ~= '' then
		f = hdf5.open(opt.train, 'r')
		arg1 = f:read("arg1"):all()
		arg2 = f:read("arg2"):all()
		train = torch.Tensor(2, arg1:size(1), arg1:size(2))
		train[1] = arg1
		train[2] = arg2
		train_label = f:read("label"):all()
	end

	if opt.dev ~= '' then
		f = hdf5.open(opt.dev, 'r')
		arg1 = f:read("arg1"):all()
		arg2 = f:read("arg2"):all()
		dev = torch.Tensor(2, arg1:size(1), arg1:size(2))
		dev[1] = arg1
		dev[2] = arg2
		dev_label = f:read("label"):all()
	end

	if opt.test ~= '' then
		f = hdf5.open(opt.test, 'r')
		arg1 = f:read("arg1"):all()
		arg2 = f:read("arg2"):all()
		test = torch.Tensor(2, arg1:size(1), arg1:size(2))
		test[1] = arg1
		test[2] = arg2
		test_label = f:read("label"):all()
	end

	
	print('data loaded!')

  return train, train_label, dev, dev_label, test, test_label, w2v
  -- return dev, dev_label, train, train_label, test, test_label, w2v
end

function train_model(train_data, train_label, model, criterion, layers)
	local optim_method = optim.adadelta
	local params, grads = model:getParameters()
	local state = {}
	-- print(model:getParameters():size())
	model:training()

	local train_size = train_data:size(2)
	local timer = torch.Timer()
	local time = timer:time().real
	local total_err = 0

	local classes = {}
	for i = 1, opt.num_classes do
		table.insert(classes, i)
	end
	local confusion = optim.ConfusionMatrix(classes)
	confusion:zero()

	local config -- for optim
	config = { rho = 0.95, eps = 1e-6 } 

	-- shuffle batches
	local num_batches = math.floor(train_size / opt.batch_size)
	local shuffle = torch.randperm(num_batches)
	for i = 1, shuffle:size(1) do
		local t = (shuffle[i] - 1) * opt.batch_size + 1
		local batch_size = math.min(opt.batch_size, train_size - t + 1)

		-- data samples and labels, in mini batches.
		local inputs = train_data:narrow(2, t, batch_size)
		local targets = train_label:narrow(1, t, batch_size)
		_, targets = targets:max(2)
		targets = targets:reshape(opt.batch_size)
		if opt.cudnn == 1 then
			inputs = inputs:cuda()
			targets = targets:cuda()
		else
			inputs = inputs:double()
			targets = targets:double()
		end

		-- closure to return err, df/dx
		local func = function(x)
			-- get new parameters
			if x ~= params then
			params:copy(x)
			end
			-- reset gradients
			grads:zero()

			-- forward pass
			local outputs = model:forward({inputs[1],inputs[2]})
			-- print(layers["W_hh"].output)
			local err = criterion:forward(outputs, targets)

			-- track errors and confusion
			total_err = total_err + err * batch_size
			for j = 1, batch_size do
				confusion:add(outputs[j], targets[j])
			end
			-- compute gradients
			local df_do = criterion:backward(outputs, targets)
			model:backward({inputs[1], inputs[2]}, df_do)

			-- layers.w2v.gradWeight:zero()
			return err, grads
		end

		-- gradient descent
		optim_method(func, params, config, state)
		-- reset padding embedding to zero
		-- layers.w2v.weight[1]:zero()
		-- Renorm (Euclidean projection to L2 ball)
		local renorm = function(row)
			local n = row:norm()
			row:mul(opt.L2s):div(1e-7 + n)
		end
		-- renormalize linear row weights
		-- local w = layers.linear.weight
		-- for j = 1, w:size(1) do
		-- 	renorm(w[j])
		-- end
	end
	if opt.debug == 1 then
		print('Tranin perf : ' .. total_err)
		print(confusion)
	end
	-- time taken
	time = timer:time().real - time
	time = opt.batch_size * time / train_size
	if opt.debug == 1 then
		print("==> time to learn 1 batch = " .. (time*1000) .. 'ms')
	end
	-- return error percent
	confusion:updateValids()
	return confusion.totalValid

end

function test_model(test_data, test_label, model, criterion, layers)
	model:evaluate()

	local classes = {}
	for i = 1, opt.num_classes do
		table.insert(classes, i)
	end
	local confusion = optim.ConfusionMatrix(classes)
	confusion:zero()
	local test_size = test_data:size(2)
	local total_acc = 0
	--  detailed info.
	-- local error_detail = {}
	-- local idx_to_word = {}
	-- local word_maping_file = "custom_word_mapping.txt"
	-- for line in io.lines(word_maping_file) do
	-- 	local pair = {}
	-- 	for v in line:gmatch("%S+") do
	-- 		pair[#pair+1] = v
	-- 	end
	-- 	table.insert(idx_to_word, pair[1])
	-- end
	-- local idx_to_class = {'space','power','operation','oilwear','comfort','appearance','decoration','costperformance', 'failure', 'maintenance', 'neutral'}

	for t = 1, test_size, opt.batch_size do
	-- data samples and labels, in mini batches.
		local batch_size = math.min(opt.batch_size, test_size - t + 1)
		local inputs = test_data:narrow(2, t, batch_size)
		local targets = test_label:narrow(1, t, batch_size)
		if opt.cudnn == 1 then
			inputs = inputs:cuda()
			targets = targets:cuda()
		else
			inputs = inputs:double()
			targets = targets:double()
		end
		local outputs = model:forward({inputs[1], inputs[2]})
		for i = 1, batch_size do
			confusion:add(outputs[i], targets[i])
			-- output error sentences.
			local _, predict = outputs[i]:max(1)
			-- if predict[1] ~= targets[i] then
			if targets[i][predict[1]] == 2 then
				total_acc = total_acc + 1
				-- local sen = ""
				-- for j = 1, inputs[i]:size(1) do
				--   if inputs[i][j] ~= 1 then
				--     sen = sen .. idx_to_word[inputs[i][j]]
				--   end
				-- end
				-- local e = {idx_to_class[targets[i]], idx_to_class[predict[1]], sen}
				-- table.insert(error_detail, e)
				-- print(table.getn(error_detail))
			else
			end
		end
	end

	if opt.debug == 1 then
		print(confusion)
		-- f = io.open("results/error_detail.out", "w")
		-- for k,e in pairs(error_detail) do
		-- 	f:write("pred: " .. e[2] .. '\n' .. "true: " .. e[1] .. '\n' .. e[3] .. '\n\n')
		-- end
	end

	-- return error percent
	confusion:updateValids()
	-- print(confusion.totalValid)
	-- print(total_acc/test_size)
	return total_acc / test_size
end

function save_model(model) 
	if not path.exists('results') then lfs.mkdir('results') end

	local savefile
	if opt.savefile ~= '' then
		savefile = opt.savefile
	else
		savefile = string.format('results/%s_model.t7', os.date('%Y%m%d_%H%M'))
	end
	print('saving results to ', savefile)
	local save = {}
	save['opt'] = opt
	save['model'] = model
	torch.save(savefile, save)
end


function main()

	torch.setnumthreads(1)

	-- parse arguments
	opt = cmd:parse(arg)

	local train, train_label
	local test, test_label
	local dev, dev_label
	local w2v

	train, train_label, dev, dev_label, test, test_label, w2v = load_data(w2v)
	opt.vocab_size = w2v:size(1)
	opt.vec_size = w2v:size(2)
	loadstring("opt.config = " .. opt.config)()
	print(opt.config)
	print(opt.vocab_size)

	if opt.train_only == 1 then
		
		opt.max_sent = train[1]:size(2)
		opt.num_classes = train_label:size(2)
		local best_model, best_epoch
		local best_perf = 0.0 -- save best model

		local timer = torch.Timer()
		local start_time = timer:time().real

		local model, criterion, layers = build_model(w2v)

		local res_file = io.open("train_res.out", "w")

		for epoch = 1, opt.epochs do
			local epoch_time = timer:time().real
			local train_perf = 1.0
			local train_perf = train_model(train, train_label, model, criterion, layers)
			-- No early stopping

			local dev_perf = test_model(dev, dev_label, model, criterion, layers)
			print('epoch:', epoch, 'train perf:', 100*train_perf, '%, val perf: ', 100*dev_perf, '%')
			res_file:write(epoch .. '\t' .. string.format("%.2f",100*train_perf) .. '\t' ..  string.format("%.2f",100*dev_perf) .. '\n')
			if dev_perf >= best_perf then
				best_model = model:clone()
				best_perf = dev_perf
				best_epoch = epoch
			end
			if epoch % 10 == 0 then
				save_model(best_model)
			end

			-- early stopping
			-- if epoch % 5 == 0 then
			-- 	local dev_perf = test_model(dev, dev_label, model, criterion)
			-- 	print('epoch:', epoch, 'train perf:', 100*train_perf, '%, val perf: ', 100*dev_perf, '%')
			-- 	if dev_perf >= best_perf then
			-- 		best_model = model:clone()
			-- 		best_perf = dev_perf
			-- 		best_epoch = epoch
			-- 	else
			-- 		break
			-- 	end
			-- else
			-- 	print('epoch:', epoch, 'train perf:', 100*train_perf, '%')
			-- end

		end
		save_model(best_model)

	else
		opt.max_sent = test[1]:size(1)
		opt.num_classes = test_label:size(2)
		local model, criterion, layers = build_model(w2v)
		local test_err = test_model(test, test_label, model, criterion, layers)
		print('test perf: ', 100*test_err, '%')
	end

	return 
end

main()
th main.lua -train data/pdtb_train.hdf5 -dev data/pdtb_dev.hdf5 -mtype SCNN -train_only 1 -cudnn 1
th main.lua -train data/pdtb_train.hdf5 -dev data/pdtb_dev.hdf5 -mtype CNN -config "{W=5}" -train_only 1 -cudnn 1
th main.lua -train data/pdtb_train.hdf5 -dev data/pdtb_dev.hdf5 -mtype RNN -config "{rho=10}" -train_only 1 -cudnn 1
th main.lua -train data/pdtb_train.hdf5 -dev data/pdtb_dev.hdf5 -mtype TCNN -config "{}" -train_only 1 -cudnn 1

th main.lua -test data/pdtb_dev.hdf5 -model results/40.77_150d*200s.t7 -train_only 0 -cudnn 1
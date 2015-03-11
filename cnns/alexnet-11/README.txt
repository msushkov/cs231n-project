AlexNet, but with the last softmax layer replaced with an 11-class softmax (11-th class is trash).

------------------------------------------------------------

Run 1:

test_iter = 1000
test_interval = 1000
base_lr = 0.0001
gamma = 0.1
stepsize = 1000
weight_decay = 0.0005
solver_type = ADAGRAD

batch size = 256

accuracy after about 9k iterations (on val/bicycle only) = 0.75 (didn't change at all throughout the 9k iterations)

loss after 9k iterations = between 0.17 and 0.28

------------------------------------------------------------

Run 2:

test_iter = 30
test_interval = 200
base_lr = 0.0005
gamma = 0.1
stepsize = 500
weight_decay = 0.0005
solver_type = ADAGRAD

batch size = 100

accuracy after about 5400 iterations (on val/5 classes) = 0.726
iter acc
800 0.731
1000 0.737
2000 0.7407
3000 0.737
4000 0.748
5000 0.729

iter loss
740 0.10723
2000 0.181
3000 0.099
4000 0.131
5000 0.161
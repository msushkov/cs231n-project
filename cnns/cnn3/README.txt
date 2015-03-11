Simple 3-layer CNN

-----------------------------------------

net: "/root/cs231n-project/cnns/cnn3/train_val.prototxt"
test_iter: 30
test_interval: 200
base_lr: 0.01
lr_policy: "step"
gamma: 0.5
stepsize: 200
display: 100
max_iter: 450000
weight_decay: 0.0005
snapshot: 500
snapshot_prefix: "/root/cs231n-project/cnns/cnn3/snapshots/cnn3"
solver_type: ADAGRAD

200 iters: acc = 0.666, loss = 2.11
800 iters: acc = 0.662, loss = 1.85
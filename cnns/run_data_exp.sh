#!/usr/bin/env sh

dir=$(pwd)

cd googlenet/train_all_layers/data_experiments

echo "no_aug_IM_IG..."
cd no_aug_IM_IG
source train_net.sh

echo "aug_1_IM..."
cd ../aug_1_IM
source train_net.sh

echo "aug_1_IM_IG"
cd ../aug_1_IM_IG
source train_net.sh

echo "aug_2_IM"
cd ../aug_2_IM
source train_net.sh

echo "aug_2_IM_IG"
cd ../aug_2_IM_IG
source train_net.sh

echo "Done."
cd $dir
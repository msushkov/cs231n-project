#!/usr/bin/env sh

dir=$(pwd)

echo "alexnet-11/fix_all_other_layers..."
cd alexnet-11/fix_all_other_layers
source train_net.sh

echo "alexnet-11/train_all_layers..."
cd ../train_all_layers
source train_net.sh

echo "googlenet/fix_all_other_layers..."
cd ../../googlenet/fix_all_other_layers
source train_net.sh

echo "googlenet/train_all_layers..."
cd ../train_all_layers
source train_net.sh

echo "nin/fix_all_other_layers..."
cd ../../nin/fix_all_other_layers
source train_net.sh

echo "nin/train_all_layers..."
cd ../train_all_layers
source train_net.sh

echo "Done."
cd $dir
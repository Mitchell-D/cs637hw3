#!/usr/bin/bash

#epochs=14
epochs=25
imgdir="figures/activations/"
outdir="figures/cifar10_L2/"
#outdir="figures/mnist_L2/"
dataset="cifar10"
layer=2
tilesize="64x64"
gridsize="4x8"

for epoch in $(seq -f "%02g" 0 `expr $epochs - 1`);
do
    montage -border 0 -geometry $tilesize -tile $gridsize \
        $imgdir$dataset"_L"$layer"-E"$epoch"-K*.png" \
        $outdir$dataset"_L"$layer"_E"$epoch".png"
done

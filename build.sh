#!/usr/bin/env bash

# Accept MODEL and DATASET from environment, default to minivit/mnist
MODEL=${MODEL:-minivit}
DATASET=${DATASET:-mnist}
ELEM_T_IS_FLOAT=${ELEM_T_IS_FLOAT:-0}

# Determine the suffix based on quantization
if [ "$ELEM_T_IS_FLOAT" -eq 1 ]; then
    SUFFIX=params
else
    SUFFIX=quantized_params
fi

WEIGHTS_HEADER="includes/${MODEL}_${DATASET}_${SUFFIX}.h"

echo "Building with MODEL=$MODEL, DATASET=$DATASET"
echo "Using weights header: $WEIGHTS_HEADER"

# Export as CFLAGS to pass to make
export EXTRA_CFLAGS="-DMODEL=$MODEL -DDATASET=$DATASET -DWEIGHTS_HEADER=\"$WEIGHTS_HEADER\""

if [ ! -d "build" ] ; then
    autoconf && \
        mkdir build && cd build && \
        ../configure &&
        cd ..

    if [ $? -ne 0 ] ; then
        echo $0 failed
        exit 1
    fi
fi

cd build

# Force rebuild of vit_merged to pick up new MODEL/DATASET
if [ -f "ViTRuns/vit_merged-baremetal" ]; then
    echo "Cleaning previous build..."
    rm -f ViTRuns/vit_merged-baremetal ViTRuns/vit_merged-linux ViTRuns/vit_merged-pk
fi

if [[ $(which riscv64-unknown-linux-gnu-gcc) ]] ; then
    make -j $@
else
    make -j BAREMETAL_ONLY=1 $@
fi


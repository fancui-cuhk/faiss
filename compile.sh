#!/bin/bash

rm -rf build

cmake -B build -DFAISS_ENABLE_GPU=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_ENABLE_C_API=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-undefined,dynamic_lookup" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-undefined,dynamic_lookup" \
  .

make -C build

sudo make -C build install

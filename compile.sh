#!/bin/zsh

source ~/.zshrc

rm -rf build

conda activate faiss_arm64

cmake -B build -DFAISS_ENABLE_GPU=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_ENABLE_C_API=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-undefined,dynamic_lookup" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-undefined,dynamic_lookup" \
  .

make -C build -j

sudo make -C build install

mkdir build
cd build
# Change to your pytorch cmake prefix, like /usr/local/lib/python3.8/dist-packages/torch/share/cmake
YOUR_PYTORCH_CMAKE_PREFIX= ...
cmake -DCMAKE_PREFIX_PATH=$YOUR_PYTORCH_CMAKE_PREFIX ..
make
mkdir ../lib
cp src/spproj/libspproj.so ../lib/libspproj.so
cd ..
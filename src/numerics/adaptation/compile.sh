#!/bin/bash

# TODO: This needs to be changed by the user
Mmg_dir='/home/ali/software/mmg/'

export LD_LIBRARY_PATH=$Mmg_dir/lib:$LD_LIBRARY_PATH
g++ -fPIC -shared -O3 -I$Mmg_dir/include/ mesh_adapter.cpp -L$Mmg_dir/lib \
        -Wl,-rpath=$Mmg_dir/lib -lmmg2d -o libmesh_adapter.so -std=c++17

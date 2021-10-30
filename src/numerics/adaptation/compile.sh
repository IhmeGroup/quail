#!/bin/bash

# TODO: This needs to be in LD_LIBRARY_PATH when Quail is run too, so figure out
# some way to make that happen. For now, just run this export before running
# Quail
export LD_LIBRARY_PATH=/home/ali/software/mmg/lib:$LD_LIBRARY_PATH
g++ -fPIC -shared -O3 -I/home/ali/software/mmg/include/ mesh_adapter.cpp -L/home/ali/software/mmg/lib -lmmg2d -o libmesh_adapter.so

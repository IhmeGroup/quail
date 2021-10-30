#include <cstdio>
#include <iostream>

#include "mmg/mmg2d/libmmg2d.h"

using std::cout, std::endl;

extern "C" {

void adapt_mesh(const double* node_coords) {
    cout << "Starting mesh adaptation..." << endl;


    MMG5_pMesh mmgMesh;
    MMG5_pSol mmgSol;
    mmgMesh = NULL;
    mmgSol  = NULL;
    MMG2D_Init_mesh(MMG5_ARG_start,
                    MMG5_ARG_ppMesh,&mmgMesh,MMG5_ARG_ppMet,&mmgSol,
                    MMG5_ARG_end);

}

}

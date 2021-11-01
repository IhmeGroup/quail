#include <cstdio>
#include <iostream>

#include "mmg/mmg2d/libmmg2d.h"

using std::cout, std::endl;

#define MAX0(a,b)     (((a) > (b)) ? (a) : (b))
#define MAX4(a,b,c,d)  (((MAX0(a,b)) > (MAX0(c,d))) ? (MAX0(a,b)) : (MAX0(c,d)))

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

    // Get the size of the mesh: vertices, triangles, quadrangles, edges
    if ( MMG2D_Set_meshSize(mmgMesh,4,2,0,4) != 1 )  exit(EXIT_FAILURE);

    // Give the vertices: for each vertex, give the coordinates, the reference
    // and the position in mesh of the vertex
    if ( MMG2D_Set_vertex(mmgMesh,0  ,0  ,0  ,  1) != 1 )  exit(EXIT_FAILURE);
    if ( MMG2D_Set_vertex(mmgMesh,1  ,0  ,0  ,  2) != 1 )  exit(EXIT_FAILURE);
    if ( MMG2D_Set_vertex(mmgMesh,1  ,1  ,0  ,  3) != 1 )  exit(EXIT_FAILURE);
    if ( MMG2D_Set_vertex(mmgMesh,0  ,1  ,0  ,  4) != 1 )  exit(EXIT_FAILURE);

    // Give the triangles: for each triangle,
    // give the vertices index, the reference and the position of the triangle
    if ( MMG2D_Set_triangle(mmgMesh,  1,  2,  4, 1, 1) != 1 )  exit(EXIT_FAILURE);
    if ( MMG2D_Set_triangle(mmgMesh,  2,  3,  4, 1, 2) != 1 )  exit(EXIT_FAILURE);


    // Give the edges (not mandatory): for each edge,
    // give the vertices index, the reference and the position of the edge
    if ( MMG2D_Set_edge(mmgMesh,  1,  2, 1, 1) != 1 )  exit(EXIT_FAILURE);
    if ( MMG2D_Set_edge(mmgMesh,  2,  3, 2, 2) != 1 )  exit(EXIT_FAILURE);
    if ( MMG2D_Set_edge(mmgMesh,  3,  4, 3, 3) != 1 )  exit(EXIT_FAILURE);
    if ( MMG2D_Set_edge(mmgMesh,  4,  1, 4, 4) != 1 )  exit(EXIT_FAILURE);

    // Manually set the sol
    // Give info for the sol structure: sol applied on vertex entities,
    // number of vertices=4, the sol is scalar
    if ( MMG2D_Set_solSize(mmgMesh,mmgSol,MMG5_Vertex,4,MMG5_Scalar) != 1 )
        exit(EXIT_FAILURE);

    // Give solutions values and positions
    for(int k = 1; k <= 4; k++) {
        // The value here sets the mesh density
        if ( MMG2D_Set_scalarSol(mmgSol, 0.1, k) != 1 ) exit(EXIT_FAILURE);
    }

    // (not mandatory): check if the number of given entities match with mesh size
    if ( MMG2D_Chk_meshData(mmgMesh,mmgSol) != 1 ) exit(EXIT_FAILURE);

    /** ------------------------------ STEP  II -------------------------- */
    // Remesh function
    int ier = MMG2D_mmg2dlib(mmgMesh,mmgSol);

    if ( ier == MMG5_STRONGFAILURE ) {
        fprintf(stdout,"BAD ENDING OF MMG2DLIB: UNABLE TO SAVE MESH\n");
        //return(ier);
    } else if ( ier == MMG5_LOWFAILURE )
        fprintf(stdout,"BAD ENDING OF MMG2DLIB\n");

    /** ------------------------------ STEP III -------------------------- */
    // Get results
    int np;
    int nt;
    int na;
    // Get the size of the mesh: vertices, tetra, triangles, quadrangles, edges
    if ( MMG2D_Get_meshSize(mmgMesh,&np,&nt,NULL,&na) !=1 )  exit(EXIT_FAILURE);

    // Table to know if a vertex/tetra/tria/edge is required
    int* required = (int*)calloc(MAX4(np,0,nt,na)+1 ,sizeof(int));
    if (!required) {
        perror("  ## Memory problem: calloc");
        exit(EXIT_FAILURE);
    }

    int nreq = 0;
    int ref;
    int Tria[3];
    printf("\nTriangles\n%d\n",nt);
    for(int k = 1; k <= nt; k++) {
        // Triangles recovering
        if (MMG2D_Get_triangle(mmgMesh, &(Tria[0]), &(Tria[1]), &(Tria[2]),
                              &ref, &(required[k])) != 1) {
            exit(EXIT_FAILURE);
        }
        printf("%d %d %d %d \n",Tria[0],Tria[1],Tria[2],ref);
        if (required[k])  nreq++;
    }
    printf("\nRequiredTriangles\n%d\n",nreq);
    for(int k = 1; k <= nt; k++) {
        if (required[k])  printf("%d \n",k);
    }

    free(required);
    required = NULL;

    // Free the MMG2D structures
    MMG2D_Free_all(MMG5_ARG_start,
                   MMG5_ARG_ppMesh,&mmgMesh,MMG5_ARG_ppMet,&mmgSol,
                   MMG5_ARG_end);
}

}

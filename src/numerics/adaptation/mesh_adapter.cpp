#include <algorithm>
#include <cstdio>
#include <iostream>
#include <set>

#include "mmg/mmg2d/libmmg2d.h"

using std::cout, std::endl;

#define MAX0(a,b)     (((a) > (b)) ? (a) : (b))
#define MAX4(a,b,c,d)  (((MAX0(a,b)) > (MAX0(c,d))) ? (MAX0(a,b)) : (MAX0(c,d)))

extern "C" {

MMG5_Mesh* adapt_mesh(const double* node_coords, const int* node_IDs, int& np, int& nt, int& na) {
    cout << "Starting mesh adaptation..." << endl;

    MMG5_Mesh* mmgMesh = (MMG5_Mesh*) malloc(sizeof(MMG5_Mesh));
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
    if ( MMG2D_Set_edge(mmgMesh, 1, 2, 1, 1) != 1 )  exit(EXIT_FAILURE);
    if ( MMG2D_Set_edge(mmgMesh, 2, 3, 2, 2) != 1 )  exit(EXIT_FAILURE);
    if ( MMG2D_Set_edge(mmgMesh, 3, 4, 2, 3) != 1 )  exit(EXIT_FAILURE);
    if ( MMG2D_Set_edge(mmgMesh, 4, 1, 1, 4) != 1 )  exit(EXIT_FAILURE);

    // Manually set the sol
    // Give info for the sol structure: sol applied on vertex entities,
    // number of vertices=4, the sol is scalar
    if ( MMG2D_Set_solSize(mmgMesh,mmgSol,MMG5_Vertex,4,MMG5_Scalar) != 1 )
        exit(EXIT_FAILURE);

    // Give solutions values and positions
    for(int k = 1; k <= 4; k++) {
        // The value here sets the mesh density
        if (k == 1) {
            if ( MMG2D_Set_scalarSol(mmgSol, 1., k) != 1 ) exit(EXIT_FAILURE);
        } else {
            if ( MMG2D_Set_scalarSol(mmgSol, 1., k) != 1 ) exit(EXIT_FAILURE);
        }
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

    // Get sizing information
    if ( MMG2D_Get_meshSize(mmgMesh, &np, &nt, nullptr, &na) !=1 )  exit(EXIT_FAILURE);

    return mmgMesh;
}

void get_results(MMG5_pMesh mmgMesh, double* node_coords, long* node_IDs, long* face_info,
        long* num_faces_per_bgroup, long* bface_info) {


    /** ------------------------------ STEP III -------------------------- */
    // Get results
    int np;
    int nt;
    int na;
    // Get the size of the mesh: vertices, tetra, triangles, quadrangles, edges
    if (MMG2D_Get_meshSize(mmgMesh, &np, &nt, NULL, &na) !=1 )  exit(EXIT_FAILURE);

    // Table to know if a vertex/tetra/tria/edge is required
    int* required = (int*)calloc(MAX4(np,0,nt,na)+1 ,sizeof(int));
    if (!required) {
        perror("  ## Memory problem: calloc");
        exit(EXIT_FAILURE);
    }

    /* Table to know if a coponant is corner and/or required */
    int* ridge = (int*)calloc(na+1 ,sizeof(int));
    if (!ridge) {
      perror("  ## Memory problem: calloc");
      exit(EXIT_FAILURE);
    }

    // TODO: Some unfortunate 1-indexing below...fix later

    // Get vertices
    double Point[3];
    int nreq = 0;
    int ref;
    nreq = 0;
    printf("\nVertices\n%d\n", np);
    for(int k = 1; k <= np; k++) {
        /** b) Vertex recovering */
        if ( MMG2D_Get_vertex(mmgMesh, &(Point[0]), &(Point[1]),
                            &ref, NULL, &(required[k])) != 1 ) {
            exit(EXIT_FAILURE);
        }
        printf("%.15lg %.15lg %d \n", Point[0], Point[1], ref);
        // Store node coordinates
        node_coords[2*(k-1)] = Point[0];
        node_coords[2*(k-1) + 1] = Point[1];
        if (required[k])  nreq++;
    }
    printf("\nRequiredVertices\n%d\n", nreq);
    for(int k = 1; k <= np; k++) {
      if (required[k]) printf("%d \n", k);
    }

    // Get triangles
    int Tria[3];
    int neighbors[3];
    int neighbors_of_neighbors[3];
    auto num_interior_faces = ((nt * 3) - na) / 2;
    int global_face_ID = 0;
    std::set<std::pair<int, int> > created_faces;
    printf("\nTriangles\n%d\n", nt);
    for(int elem_ID = 1; elem_ID <= nt; elem_ID++) {
        // Triangles recovering
        if (MMG2D_Get_triangle(mmgMesh, &(Tria[0]), &(Tria[1]), &(Tria[2]),
                              &ref, &(required[elem_ID])) != 1) {
            exit(EXIT_FAILURE);
        }
        printf("Nodes: %d %d %d %d \n",Tria[0],Tria[1],Tria[2],ref);
        //printf("Edges: %d %d %d %d %d %d\n", mmgMesh->tria[k], 1, 1, 1, 1, 1);
        //cout << mmgMesh->tria[k].v[0] << endl;
        if (MMG2D_Get_adjaTri(mmgMesh, elem_ID, neighbors) != 1) exit(EXIT_FAILURE);
        cout << "Neighbors: " << neighbors[0] << " " << neighbors[1] << " " << neighbors[2] << endl;
        // Store node IDs, subtracting one to go back to 0-indexed
        node_IDs[3*(elem_ID - 1)] = Tria[0] - 1;
        node_IDs[3*(elem_ID - 1) + 1] = Tria[1] - 1;
        node_IDs[3*(elem_ID - 1) + 2] = Tria[2] - 1;
        // Loop over faces of this triangle
        for (int face_ID = 0; face_ID < 3; face_ID++) {
            // Get neighbor of this face
            auto neighbor_ID = neighbors[face_ID];
            // If it's a boundary face, skip it
            if (neighbor_ID == 0) continue;
            // Has this face been made already?
            bool made_yet = created_faces.find(std::pair(neighbor_ID, elem_ID)) != created_faces.end();
            // If it hasn't, then make it
            if (not made_yet) {
                // Get face ID on the neighbor's side
                if (MMG2D_Get_adjaTri(mmgMesh, neighbor_ID, neighbors_of_neighbors) != 1) exit(EXIT_FAILURE);
                auto neighbor_face_ID = std::distance(neighbors_of_neighbors, std::find(neighbors_of_neighbors, neighbors_of_neighbors + 3, elem_ID));
                // Store face information
                face_info[4 * global_face_ID] = elem_ID - 1;
                face_info[4 * global_face_ID + 1] = neighbor_ID - 1;
                face_info[4 * global_face_ID + 2] = face_ID;
                face_info[4 * global_face_ID + 3] = neighbor_face_ID;
                // Increment index of faces
                global_face_ID++;
                // Mark this face as created
                created_faces.insert(std::pair(elem_ID, neighbor_ID));
            }
        }

        if (required[elem_ID])  nreq++;
    }
    printf("\nRequiredTriangles\n%d\n",nreq);
    for(int k = 1; k <= nt; k++) {
        if (required[k])  printf("%d \n",k);
    }

    nreq = 0;
    int nr = 0;
    int Edge[2];
    int elem_IDs[2];
    int face_IDs[2];
    printf("\nEdges\n%d\n",na);
    for(int edge_idx = 1; edge_idx <= na; edge_idx++) {
        // Get the vertices of the edge as well as its reference, and whether
        // it's a ridge or required edge
        if ( MMG2D_Get_edge(mmgMesh,&(Edge[0]),&(Edge[1]),&ref,
                            &(ridge[edge_idx]),&(required[edge_idx])) != 1 )  exit(EXIT_FAILURE);
        // Get the element and face IDs on either side of the edge
        if (MMG2D_Get_trisFromEdge(mmgMesh, edge_idx, elem_IDs, face_IDs) != 1) exit(EXIT_FAILURE);
        // Store information. Since all Mmg edges here will be boundary faces,
        // only the first element/face ID will be relevant, since there is only
        // one neighbor. Also, since ref is 1-indexed, must subtract 1.
        bface_info[3 * (edge_idx - 1)] = elem_IDs[0] - 1;
        bface_info[3 * (edge_idx - 1) + 1] = face_IDs[0];
        bface_info[3 * (edge_idx - 1) + 2] = ref - 1;
        // Increment the number of faces in this group
        num_faces_per_bgroup[ref - 1]++;
        cout << "edge stuff" << endl;
        cout << edge_idx << "  " << elem_IDs[0] << " " << elem_IDs[1] << " " << face_IDs[0] << " " << face_IDs[1] << endl;

        printf("%d %d %d \n",Edge[0],Edge[1],ref);
        if ( ridge[edge_idx] )  nr++;
        if ( required[edge_idx] )  nreq++;
    }
    printf("\nRequiredEdges\n%d\n",nreq);
    for(int k=1; k<=na; k++) {
        if ( required[k] )  printf("%d \n",k);
    }

    free(required);
    required = NULL;

    // Free the MMG2D structures
    // TODO
    //MMG2D_Free_all(MMG5_ARG_start,
    //               MMG5_ARG_ppMesh,&mmgMesh,MMG5_ARG_ppMet,&mmgSol,
    //               MMG5_ARG_end);
}

}

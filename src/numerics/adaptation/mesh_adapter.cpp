#include <algorithm>
#include <cstdio>
#include <iostream>
#include <set>
#include <cmath>

#include "mmg/mmg2d/libmmg2d.h"

using std::cout, std::endl;

#define MAX0(a,b)     (((a) > (b)) ? (a) : (b))
#define MAX4(a,b,c,d)  (((MAX0(a,b)) > (MAX0(c,d))) ? (MAX0(a,b)) : (MAX0(c,d)))

void check_error(int error) {
    if (error != 1) exit(EXIT_FAILURE);
}

extern "C" {

void adapt_mesh(const double* node_coords, const long* node_IDs, const long*
        bface_info, const double* eigvals, const double* eigvecs, int&
        num_nodes, int& num_elems, int& num_edges, MMG5_pMesh& mmgMesh,
        MMG5_pSol& mmgSol) {
    int error;
    int ndims = 2;

    mmgMesh = NULL;
    mmgSol  = NULL;
    MMG2D_Init_mesh(MMG5_ARG_start,
                    MMG5_ARG_ppMesh,&mmgMesh,MMG5_ARG_ppMet,&mmgSol,
                    MMG5_ARG_end);

    // -- Set Mmg global parameters -- //
    // hgrad: ratio of connected edge lengths. Default is 1.3.
    error = MMG2D_Set_dparameter(mmgMesh, mmgSol, MMG2D_DPARAM_hgrad, 2.);
    check_error(error);

    // Set the size of the mesh: vertices, triangles, quadrangles, and edges
    error = MMG2D_Set_meshSize(mmgMesh, num_nodes , num_elems, 0, num_edges);
    check_error(error);

    // Loop over nodes
    for (int node_ID = 0; node_ID < num_nodes; node_ID++) {
        // Set the vertices. For each vertex, give the coordinates, the reference
        // and the index (with 1-indexing).
        error = MMG2D_Set_vertex(mmgMesh, node_coords[2*node_ID],
                node_coords[2*node_ID + 1], 0, node_ID + 1);
        check_error(error);
        // Set vertex requirements
        // TODO: How best to handle this in general?
        error = MMG2D_Set_corner(mmgMesh, node_ID + 1);
        check_error(error);
    }

    // Loop over elements
    for (int elem_ID = 0; elem_ID < num_elems; elem_ID++) {
        // Set the triangles. For each triangle, give the vertex indices, the
        // reference and the index of the triangle (with 1-indexing).
        error = MMG2D_Set_triangle(mmgMesh, node_IDs[3*elem_ID] + 1,
                node_IDs[3*elem_ID + 1] + 1, node_IDs[3*elem_ID + 2] + 1, 1,
                elem_ID + 1);
        check_error(error);
    }

    // Loop over edges
    for (int edge_ID = 0; edge_ID < num_edges; edge_ID++) {
        // Set the edges. For each edge, give the vertex indices (with
        // 1-indexing), the reference (with 1-indexing) and the index of the
        // edge (with 1-indexing).
        auto info = bface_info + 3*edge_ID;
        error = MMG2D_Set_edge(mmgMesh, info[0] + 1, info[1] + 1, info[2] + 1,
                edge_ID + 1);
        check_error(error);
    }
    // Set edge requirements
    // TODO: How best to handle this in general?
    //error = MMG2D_Set_requiredEdge(mmgMesh, 1);
    //error = MMG2D_Set_requiredEdge(mmgMesh, 2);
    //check_error(error);

    // Give info to the sol struct.
    // - the mesh and sol structs
    // - where the sol will be applied, which is the vertices
    // - the number of vertices
    // - the type of sol, which is scalar for now (isotropic)
    error = MMG2D_Set_solSize(mmgMesh, mmgSol, MMG5_Vertex, num_nodes, MMG5_Tensor);
    check_error(error);

    // Give sol values and positions
    auto c = 1.28057912;
    for (int k = 1; k <= num_nodes; k++) {
        // The value here sets the mesh density
        auto coords = node_coords + 2*(k-1);
        auto amp = 3*pow(coords[1] - c * coords[0], 2) + .05;
        auto lambda1 = 1/(amp * amp * 1e-2);
        auto lambda2 = 1/(amp * amp);
        // For elements far from the shock, or with tiny gradients
        double limit = 1e4;
        if (amp > .05 or (eigvals[(k-1)*ndims] < limit) or (eigvals[(k-1)*ndims + 1] < limit)) {
            amp = 10.;
            // Set isotropic metric
            error = MMG2D_Set_tensorSol(mmgSol, 1/(amp * amp), 0., 1/(amp * amp), k);
        // For elements near the shock
        } else {
            // matrix multiply with eigenvectors
            const double* V = eigvecs + (k-1)*ndims*ndims;
            const double V_T[4] = {V[0], V[2], V[1], V[3]};
            double V_lambda[4] = {V[0] * lambda1, V[1] * lambda2, V[2] * lambda1, V[3] * lambda2};
            double V_lambda_V_T[4] = {V_lambda[0] * V_T[0] + V_lambda[1] * V_T[2],
                                      V_lambda[0] * V_T[1] + V_lambda[1] * V_T[3],
                                      V_lambda[2] * V_T[0] + V_lambda[3] * V_T[2],
                                      V_lambda[2] * V_T[1] + V_lambda[3] * V_T[3]};
            // Set anisotropic metric
            error = MMG2D_Set_tensorSol(mmgSol, V_lambda_V_T[0], V_lambda_V_T[1], V_lambda_V_T[3], k);
        }
        check_error(error);
    }

    // (not mandatory): check if the number of given entities match with mesh size
    error = MMG2D_Chk_meshData(mmgMesh,mmgSol);
    check_error(error);

    /** ------------------------------ STEP  II -------------------------- */
    // Remesh function
    int ier = MMG2D_mmg2dlib(mmgMesh,mmgSol);

    if ( ier == MMG5_STRONGFAILURE ) {
        fprintf(stdout,"BAD ENDING OF MMG2DLIB: UNABLE TO SAVE MESH\n");
        //return(ier);
    } else if ( ier == MMG5_LOWFAILURE )
        fprintf(stdout,"BAD ENDING OF MMG2DLIB\n");

    // Get sizing information
    error = MMG2D_Get_meshSize(mmgMesh, &num_nodes, &num_elems, nullptr,
            &num_edges);
    check_error(error);
}

void get_results(MMG5_pMesh mmgMesh, MMG5_pSol mmgSol, double* node_coords, long* node_IDs, long* face_info,
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

        printf("%d %d %d \n",Edge[0],Edge[1],ref);
        if ( ridge[edge_idx] )  nr++;
        if ( required[edge_idx] )  nreq++;
    }
    printf("\nRequiredEdges\n%d\n",nreq);
    for(int k=1; k<=na; k++) {
        if ( required[k] )  printf("%d \n",k);
    }

    free(ridge);
    free(required);

    // Free the MMG2D structures
    MMG2D_Free_all(MMG5_ARG_start,
                   MMG5_ARG_ppMesh, &mmgMesh, MMG5_ARG_ppMet, &mmgSol,
                   MMG5_ARG_end);
}

}

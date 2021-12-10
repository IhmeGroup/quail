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

void initialize(const double* node_coords, const long* node_IDs, const long*
        bface_info, const int& num_nodes, const int& num_elems, const int&
        num_edges, MMG5_pMesh& mmgMesh, MMG5_pSol& mmgSol, double* metric) {
    int error;

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
        // TODO: THIS IS VERY IMPORTANT!!!
        //error = MMG2D_Set_corner(mmgMesh, node_ID + 1);
        check_error(error);
    }

    // TODO: THIS IS VERY IMPORTANT!!!
    // Use 4 for the oblique shock, 6 for the forward step.
    int n_corners = 6;
    for (int i = 1; i <= n_corners; i++) {
        error = MMG2D_Set_corner(mmgMesh, i);
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

    // Give info to the sol struct.
    // - the mesh and sol structs
    // - where the sol will be applied, which is the vertices
    // - the number of vertices
    // - the type of sol, which is scalar for now (isotropic)
    error = MMG2D_Set_solSize(mmgMesh, mmgSol, MMG5_Vertex, num_nodes, MMG5_Tensor);
    check_error(error);

    // Compute the isotropic metric from the current mesh
    error = MMG2D_doSol(mmgMesh, mmgSol);
    check_error(error);
    // Store
    double m11, m12, m22;
    for (int node_ID = 0; node_ID < num_nodes; node_ID++) {
        error = MMG2D_Get_ithSol_inSolsAtVertices(mmgSol, 0, &m11, node_ID + 1);
        error = MMG2D_Get_ithSol_inSolsAtVertices(mmgSol, 1, &m12, node_ID + 1);
        error = MMG2D_Get_ithSol_inSolsAtVertices(mmgSol, 2, &m22, node_ID + 1);
        metric[node_ID * 4] = m11;
        metric[node_ID * 4 + 1] = m12;
        metric[node_ID * 4 + 2] = m12;
        metric[node_ID * 4 + 3] = m22;
    }
}

void adapt_mesh(const double* metric, int& num_nodes, int& num_elems, int&
        num_edges, MMG5_pMesh& mmgMesh, MMG5_pSol& mmgSol) {
    int error;
    int ndims = 2;

    // Give sol values and positions
    for (int k = 1; k <= num_nodes; k++) {
        auto metric_k = metric + (k-1)*ndims*ndims;
        // Set metric
        error = MMG2D_Set_tensorSol(mmgSol, metric_k[0], metric_k[1], metric_k[3], k);
        check_error(error);
    }

    // (not mandatory): check if the number of given entities match with mesh size
    error = MMG2D_Chk_meshData(mmgMesh, mmgSol);
    check_error(error);

    /** ------------------------------ STEP  II -------------------------- */
    // Remesh function
    int ier = MMG2D_mmg2dlib(mmgMesh, mmgSol);

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

    /* Table to know if a component is corner and/or required */
    int* ridge = (int*)calloc(na + 1, sizeof(int));
    if (!ridge) {
      perror("  ## Memory problem: calloc");
      exit(EXIT_FAILURE);
    }

    // Get vertices
    double Point[3];
    int nreq = 0;
    int ref;
    nreq = 0;
    for(int k = 1; k <= np; k++) {
        /** b) Vertex recovering */
        if ( MMG2D_Get_vertex(mmgMesh, &(Point[0]), &(Point[1]),
                            &ref, NULL, &(required[k])) != 1 ) {
            exit(EXIT_FAILURE);
        }
        // Store node coordinates
        node_coords[2*(k-1)] = Point[0];
        node_coords[2*(k-1) + 1] = Point[1];
        if (required[k])  nreq++;
    }

    // Get triangles
    int Tria[3];
    int neighbors[3];
    int neighbors_of_neighbors[3];
    auto num_interior_faces = ((nt * 3) - na) / 2;
    int global_face_ID = 0;
    std::set<std::pair<int, int> > created_faces;
    for(int elem_ID = 1; elem_ID <= nt; elem_ID++) {
        // Triangles recovering
        if (MMG2D_Get_triangle(mmgMesh, &(Tria[0]), &(Tria[1]), &(Tria[2]),
                              &ref, &(required[elem_ID])) != 1) {
            exit(EXIT_FAILURE);
        }
        if (MMG2D_Get_adjaTri(mmgMesh, elem_ID, neighbors) != 1) exit(EXIT_FAILURE);
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

    nreq = 0;
    int nr = 0;
    int Edge[2];
    int elem_IDs[2];
    int face_IDs[2];
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

        if ( ridge[edge_idx] )  nr++;
        if ( required[edge_idx] )  nreq++;
    }

    free(ridge);
    free(required);

    // Free the MMG2D structures
    MMG2D_Free_all(MMG5_ARG_start,
                   MMG5_ARG_ppMesh, &mmgMesh, MMG5_ARG_ppMet, &mmgSol,
                   MMG5_ARG_end);
}

double edge_sign(double* p1, double* p2, double* p3) {
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
}

// Check if a point p is inside a triangle given by vertices v1, v2, v3
bool point_in_triangle(double* p, double* v1, double* v2, double* v3) {
    auto d1 = edge_sign(p, v1, v2);
    auto d2 = edge_sign(p, v2, v3);
    auto d3 = edge_sign(p, v3, v1);

    auto has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0);
    auto has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0);

    return not (has_neg and has_pos);
}

int search_mesh(double* p, double* node_coords, int num_elems) {
    int elem_ID_found = -1;
    // Loop over elements
    for (int elem_ID = 0; elem_ID < num_elems; elem_ID++) {
        auto v1 = node_coords + elem_ID*6;
        auto v2 = node_coords + elem_ID*6 + 2;
        auto v3 = node_coords + elem_ID*6 + 4;
        if (point_in_triangle(p, v1, v2, v3)) {
            elem_ID_found = elem_ID;
            break;
        }
    }
    return elem_ID_found;
}


}

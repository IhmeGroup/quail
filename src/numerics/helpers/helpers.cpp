#include <iostream>

extern "C" {

// Compute element average state.
void get_element_mean(const double* Uq, const double* w, const double* dJ, const double* vol,
        double* U_mean, int ne, int nq, int ns){

// Loop elements in parallel
// #pragma omp parallel for
    for (int i = 0; i < ne; i++) {
        // Sum over quadrature points
        for (int j = 0; j < nq; j++) {
            // Sum over w, dJ, 1/vol first, since they don't depend on k
            double a = w[j] * dJ[i*nq + j] / vol[i];
            // Loop state variables and continue sum
            for (int k = 0; k < ns; k++) {
                U_mean[i*ns + k] += Uq[i*nq*ns + j*ns + k] * a;
            }
        }
    }
}


// Evaluate the face state based on the given basis values.
void evaluate_face_state(const double* Uc, const double* basis_val, 
        double* Uq, int ne, int nq, int nb, int ns, int basis_dim, 
        bool skip_interp){

// #pragma omp parallel for
    for (int i = 0; i < ne; i++) {
        // Loop over quadrature points
        for (int iq = 0; iq < nq; iq++){
            // Sum over basis
            for (int ib = 0; ib < nb; ib++) {
                double phi = basis_val[i*nb*nq + iq*nb + ib];
                for (int k = 0; k < ns; k++){
                    Uq[i*nq*ns + iq*ns + k] += phi * Uc[i*nb*ns + ib*ns + k];
                }
            }
        }
    }
} 

// Evaluate the element state based on the given basis values.
void evaluate_elem_state(const double* Uc, const double* basis_val, 
        double* Uq, int ne, int nq, int nb, int ns, int basis_dim, 
        bool skip_interp){

// #pragma omp parallel for
    for (int i = 0; i < ne; i++) {
        // Loop over quadrature points
        for (int iq = 0; iq < nq; iq++){
            // Sum over basis
            for (int ib = 0; ib < nb; ib++) {
                double phi = basis_val[iq*nb + ib];
                for (int k = 0; k < ns; k++){
                    Uq[i*nq*ns + iq*ns + k] += phi * Uc[i*nb*ns + ib*ns + k];
                }
            }
        }
    }

} // end evaluate_state
} // end extern "C"

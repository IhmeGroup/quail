#include <iostream>

extern "C" {

// Compute element average state.
void get_element_mean(const double* Uq, const double* w, const double* dJ, const double* vol,
        double* U_mean, int ne, int nq, int ns) {

// Loop elements in parallel
#pragma omp parallel for
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
        double* Uq, int nf, int nq, int nb, int ns, int basis_dim,
        bool skip_interp) {

// Loop faces in parallel
#pragma omp parallel for
    for (int i = 0; i < nf; i++) {
        const double* phi = basis_val + i*nq*nb;
        const double* Uci = Uc + i*nb*ns;
        double* Uqi = Uq + i*nq*ns;
        // Loop over quadrature points
        for (int j = 0; j < nq; j++) {
            // Loop over state variables
            for (int k = 0; k < ns; k++) {
                // Sum over bases
                for (int n = 0; n < nb; n++) {
                    Uqi[j*ns + k] += phi[j*nb + k] * Uc[n*ns + k];
                }
            }
        }
    }
}

// Evaluate the element state based on the given basis values.
void evaluate_elem_state(const double* Uc, const double* basis_val,
        double* Uq, int ne, int nq, int nb, int ns, int basis_dim,
        bool skip_interp) {

// Loop elements in parallel
#pragma omp parallel for
    for (int i = 0; i < ne; i++) {
        const double* Uci = Uc + i*nb*ns;
        double* Uqi = Uq + i*nq*ns;
        // Loop over quadrature points
        for (int j = 0; j < nq; j++) {
            // Loop over state variables
            for (int k = 0; k < ns; k++) {
                // Sum over bases
                for (int n = 0; n < nb; n++) {
                    Uqi[j*ns + k] += basis_val[j*nb + k] * Uc[n*ns + k];
                }
            }
        }
    }
} // end evaluate_state

} // end extern "C"

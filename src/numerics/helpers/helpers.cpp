#include <iostream>

extern "C" {

// Compute element average state.
void evaluate_state(const double* Uc, const double* basis_val, 
        double* Uq, int ne, int nq, int ns, int basis_dim, 
        bool skip_interp){

    // This is equivalent to
    // Uq = np.einsum('ijn, ink -> ijk', basis_val, Uc)


    // Loop elements in parallel
#pragma omp parallel for
    if (skip_interp == True) {
        Uq = Uc;
    }
    else{
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

}

}

// # def evaluate_state(Uc, basis_val, skip_interp=False):
// #   '''
// #   This function evaluates the state based on the given basis values.

// #   Inputs:
// #   -------
// #       Uc: state coefficients [ne, nb, ns]
// #       basis_val: basis values [ne, nq, nb]
// #       skip_interp: if True, then will simply copy the state coefficients;
// #           useful for a colocated scheme, i.e. quadrature points and
// #           solution nodes (for a nodal basis) are the same

// #   Outputs:
// #   --------
// #       Uq: values of state [ne, nq, ns]
// #   '''
// #   if skip_interp:
// #       Uq = Uc.copy()
// #   else:
// #       if basis_val.ndim == 3:
// #           # For faces, there is a different basis_val for each face
// #           Uq = np.einsum('ijn, ink -> ijk', basis_val, Uc)
// #       else:
// #           # For elements, all elements have the same basis_val
// #           Uq = np.einsum('jn, ink -> ijk', basis_val, Uc)

// #   return Uq # [ne, nq, ns]

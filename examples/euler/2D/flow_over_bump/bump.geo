/*** Domain parameters ***/
// Bump is centered at (0,0)
Ly = 0.5; // domain length in y-direction
Lx = 2*Ly; // domain length in x-direction
xstart = -Lx/2;
xend = Lx/2;

// Bump parameters (cosine)
A = 0.0025; // half-amplitude
lam = 0.5; // wavelength
k = 2*Pi/lam; // frequency


/*** Mesh parameters ***/
s = 1; // multiplication factor
nx_flat = 2*s; // number of points in x direction ahead of and behind bump
nx_bump = 3*s; // number of points in x direction along bump
ny = 3*s; // number of points in y direction
ls = 1; // characteristic size (doesn't matter)


/*** Create geometry ***/
Point(1) = {xstart, 0, 0, ls};
Point(2) = {xstart, Ly, 0, ls};
Point(3) = {xend, Ly, 0, ls};
Point(4) = {xend, 0, 0, ls};
Point(5) = {-lam/2, 0, 0, ls};
Point(6) = {lam/2, 0, 0, ls};
Point(7) = {-lam/2, Ly, 0, ls};
Point(8) = {lam/2, Ly, 0, ls};
Line(1) = {5, 1};
Line(2) = {1, 2};
Line(3) = {2, 7};
Line(4) = {7, 8};
Line(5) = {8, 3};
Line(6) = {3, 4};
Line(7) = {4, 6};

// Bump profile
pList[0] = 5;
nbump = 1001;
dx = lam/(nbump-1); // correction
x = -lam/2;
For i In {1: nbump-2}
	x = x + dx;
	pList[i] = newp; // next available point number
	y = A + A*Cos(k*x);
	// Add to point list
	Point(pList[i]) = {x, y, 0, ls};
EndFor
pList[nbump-1] = 6;
// Create spline
Spline(newl) = pList[];

Line(9) = {5, 7};
Line(10) = {8, 6};
Curve Loop(1) = {2, 3, -9, 1};
Plane Surface(1) = {1};
Curve Loop(2) = {9, 4, 10, -8};
Plane Surface(2) = {2};
Curve Loop(3) = {10, -7, -6, -5};
Plane Surface(3) = {3};
Reverse Surface{1, 2}; // Need this for positive Jacobian determinants


/*** Meshing ***/
// Wall-normal
Transfinite Curve {2, 9, -10, -6} = ny Using Progression 2.5;
// Streamwise, left of bump
Transfinite Curve {1, -3} = nx_flat Using Progression 1.5;
// Streamwise, right of bump
Transfinite Curve {5, -7} = nx_flat Using Progression 1.5;
// Streamwise, bump
Transfinite Curve {4, 8} = nx_bump Using Progression 1;
Transfinite Surface {1, 2, 3};
Recombine Surface {1, 2, 3}; // recombine triangles into quadrilaterals


/*** Boundaries ***/
Physical Surface("domain") = {1, 2, 3};
Physical Curve("inflow") = {2};
Physical Curve("top") = {3, 4, 5};
Physical Curve("bottom") = {1, 8, 7};
Physical Curve("outflow") = {6};

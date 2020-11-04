
lc = 1e-1; 


// First three arguments are point coordinates
// Last argument is characteristic element size
Point(1) = {-1, 0, 0, lc};
Point(2) = {1, 0,  0, 2*lc} ;



// Don't change below
Line(1) = {1,2} ;

Physical Point("Left") = {1};
//+
Physical Point("Right") = {2};
//+
Physical Curve("domain") = {1};

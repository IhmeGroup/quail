
lc = 1e-1;
Lx = 5.5; Ly = 1.1; Lz = 1.0;
Point(1) = {-1, 0, 0, lc};
Point(2) = {1, 0,  0, lc} ;

Line(1) = {1,2} ;

// Streamwise
Transfinite Line{1} = 3; //+
Physical Point("Left") = {1};
//+
Physical Point("Right") = {2};
//+
Physical Curve("domain") = {1};

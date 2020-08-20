
lc = 1.0;
nElem = 5;
L = 5.0;
Point(1) = {-L, -L, 0, lc};
Point(2) = {L, -L,  0, lc} ;
Point(3) = {L, L, 0, lc} ;
Point(4) = {-L, L, 0, lc} ;

Line(1) = {1,2} ;
Line(2) = {3,2} ;
Line(3) = {3,4} ;
Line(4) = {4,1} ;

Line Loop(1) = {4,1,-2,3} ;
Plane Surface(1) = {1} ;

// Streamwise
Transfinite Line{1,-3} = nElem+1; //51;

// Spanwise
Transfinite Line{2,4} = nElem+1; //26;
Transfinite Surface {1};
// Recombine Surface {1};

/*Extrude {0, 0, Lz} {
  Surface{1}; Layers{2}; Recombine;
}
Reverse Surface{1};*/

Physical Line("x1") = {2} ;
Physical Line("x2") = {4} ;
Physical Line("y1") = {1} ;
Physical Line("y2") = {3} ;
Physical Surface("domain") = {1} ;
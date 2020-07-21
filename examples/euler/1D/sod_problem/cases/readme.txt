-------------------------------------------------------------------------------

							Sod Problem Tests
Bornhoft, 2020
-------------------------------------------------------------------------------

This directory shows all the variations attempted with the sod shock problem. 
The cases here stem from both Yu's 2015 JCP paper on the EBDG scheme as well 
as Wang and Shu's 2012 JCP paper on PPL with detonations.

-------------------------------------------------------------------------------

------------------------------ CASES ---------------------------------

case_0: Standard DGP2, no active limiter, Yu's case

case_1: Standard DGP2, no active limiter, Shu's case 

case_2: BASE: case_1 -> Change NumTimeSteps from 2000 -> 4000
	Notes: No change from case_1

case_3: BASE: case_0 -> Change PPLimiter from OFF -> ON
	Notes: Minimized oscillations and overshoots -> NEW BASE: case_3 for Yv's case

case_4: BASE: case_1 -> Change NumTimeSteps from 2000 -> 1500
	Notes: Identical to case_1 -> NEW BASE: case_4 for Shu's case

case_5: BASE: case_4 -> Change PPLimiter from OFF -> ON for Shu's case
	Notes: Minimized oscillations and overshoots -> NEW BASE: case_5 for Shu's case

case_6: BASE: case_3 -> Change NodeType and Quadrature both to GaussLobatto (NodesEqualQuadpts=FALSE)

case_7: BASE: case_3 -> Change NodeType and Quadrature both to GaussLobatto (NodesEqualQuadpts=TRUE)
	Notes: Wierd behavior near expansion fan ... non-physical embedded shock ... maybe check ICs

case_8 BASE: case_4 -> Change NodeType and Quadrature both to GaussLobatto (NodesEqualQuadpts=TRUE)
	Notes: Wierd behavior near expansion fan ... think it has to do with IC's.

case_9 BASE: case_3 -> Change from P2 -> P29
	Notes: Extremely oscillatory

------------------------------ STUDIES ---------------------------------

limiter_yu_case: Comparison of case_0 and case_3 -> case_3 shows improvement due to limiter







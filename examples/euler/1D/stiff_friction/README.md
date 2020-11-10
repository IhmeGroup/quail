This case takes a simple advecting density wave and adds friction to the
system. The amount of friction is determined by the parameter `nu`. When
it is large, the system is stiff. 
  -  In `stiff_friction.py` you can try several different schemes and
  observe their effect on the energy. Try `SSPRK3`, `Simpler`, and 
  `ADER`. (Note: For `ADER`, uncomment the `"Solver" : "ADERDG"` 
  parameter).
  -  You can also modify how the source term is treated in the ADERDG 
  scheme. This can be done by uncommenting `SourceTreatmentADER` and 
  running it `Implicit`. By default, it is `Explicit`.
  -  What does each case calculate for the steady-state energy? What
  should it be?
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
  - After running each case, create an animation to observe the 
  simulation dynamics with the following command

```sh
$ quail -p create_anim.py
```


<p align="center">
  <img alt="anim.mp4" src="https://user-images.githubusercontent.com/55554103/98711814-5705b800-233a-11eb-893c-b272f39bc349.gif" width="50%"></a>
</p>


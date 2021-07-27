This case solves an ODE of the form dU/dt = M(U) + S(U) where M and S are mixing and reacting source term models respectively. 
  - Solutions are modeled in 1D with a single element

Two cases are provided in this input deck. 'Case A' models an extinction event and 'Case B' models an ignition event. Both cases are known to provide issues for the common Strang splitting procedure. Here, we compare Strang splitting, ADERDG, and a reference solution obtained using LSODA. 

-To use Case A or B just comment out the appropriate settings at the top of the input deck.
-To generate the solution files for the appropriate scheme comment out the other two schemes settings. After the solution is generated make sure to rename the file based on the expected file names in `post_process.py`.


<p align="center">
  <img alt="Case A" src="https://user-images.githubusercontent.com/55554103/127208044-03b1c8a6-48df-4757-b478-a0269805075e.png" width="45%"></a>
  <img alt="Case B" src="https://user-images.githubusercontent.com/55554103/127208100-c7874a1a-e000-402d-b1aa-30fc1df71537.png" width="45%"></a>
</p>

This case involves the 2D solution of a Riemann problem with a gravity source term. 
   - The specified gravity source term leads to very small values for pressure and density. These values can become negative in no limiting is applied.
   - We utilize the Positivity-Preserving Limiter (PPL) of Zhang and Shu to ensure positive physical properties.
   - Post-processing
    - Density contour with extended contour axis
    - 1D density extracted at y=1.7875

<p align="center">
  <img alt="" src="https://user-images.githubusercontent.com/55554103/104630034-1f273380-564f-11eb-8749-3864229c2f58.png" width="85%"></a>
</p>

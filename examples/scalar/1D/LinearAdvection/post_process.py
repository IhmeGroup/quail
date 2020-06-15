import code

import processing.post as Post
import processing.plot as Plot
import processing.readwritedatafiles as ReadWriteDataFiles

import solver.DG as Solver

### Postprocess
fname = "Data_final.pkl"
mesh, EqnSet, Params, Time = ReadWriteDataFiles.read_data_file(fname)
print('Solution Final Time:',Time)

solver = Solver.DG_Solver(Params,EqnSet,mesh)

# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Scalar")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver, "Scalar", PlotExact=True, PlotIC=True, Label="u")
Plot.ShowPlot()

import code

import processing.post as Post
import processing.plot as Plot
import processing.readwritedatafiles as ReadWriteDataFiles

import solver.DG as Solver

### Postprocess
fname = "Data_final.pkl"
mesh, physics, Params, Time = ReadWriteDataFiles.read_data_file(fname)
print('Solution Final Time:',Time)

solver = Solver.DG_Solver(Params,physics,mesh)
# Error
TotErr,_ = Post.L2_error(mesh, physics, solver, "Density")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, physics, solver, "Energy", PlotExact=True, PlotIC = True)

Plot.SaveFigure(FileName='SmoothIsentropicFlow', FileType='pdf', CropLevel=2)

Plot.ShowPlot()

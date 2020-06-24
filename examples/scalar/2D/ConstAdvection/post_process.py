import code

import processing.post as Post
import processing.plot as Plot
import processing.readwritedatafiles as ReadWriteDataFiles

import solver.DG as Solver
### Postprocess
# Error

### Postprocess
fname = "Data_final.pkl"
mesh, physics, Params, Time = ReadWriteDataFiles.read_data_file(fname)
print('Solution Final Time:',Time)

solver = Solver.DG(Params,physics,mesh)

solver.Time = 0.
TotErr,_ = Post.L2_error(mesh, physics, solver, "Scalar")
# Plot
axis = None
# axis = [-5., 5., -5., 5.]
Plot.PreparePlot(axis=axis, linewidth=0.5)
Plot.PlotSolution(mesh, physics, solver, "Scalar", Equidistant=True, PlotExact=False, include_mesh=True, 
	Regular2D=True, ShowTriangulation=False)
Plot.SaveFigure(FileName='Gaussian', FileType='pdf', CropLevel=2)
Plot.plot_line_probe(mesh, physics, solver, "Scalar", xy1=[-5.,-5.], xy2=[5.,5.], nPoint=101, PlotExact=True, PlotIC=True)
# Post.get_boundary_info(mesh, EqnSet, solver, "y1", "Scalar", integrate=True, 
# 		vec=[0.,1.], dot_normal_with_vec=True, plot_vs_x=True, plot_vs_y=False)
Plot.ShowPlot()

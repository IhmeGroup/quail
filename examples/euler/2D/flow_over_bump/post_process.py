import code

import processing.post as Post
import processing.plot as Plot
import processing.readwritedatafiles as ReadWriteDataFiles

import solver.DG as Solver

### Postprocess
fname = "p2_final.pkl"
mesh, physics, Params, Time = ReadWriteDataFiles.read_data_file(fname)
print('Solution Final Time:',Time)

solver = Solver.DG(Params,physics,mesh)


TotErr,_ = Post.L2_error(mesh, physics, solver, "Entropy", NormalizeByVolume=False)
# Plot
axis = None
EqualAR = False
# axis = [-5., 5., -5., 5.]
Plot.PreparePlot(axis=axis, linewidth=0.5)
Plot.PlotSolution(mesh, physics, solver, "Pressure", Equidistant=True, PlotExact=False, include_mesh=True, 
	ShowTriangulation=False, EqualAR=EqualAR, show_elem_IDs=True)
Plot.SaveFigure(FileName='Pressure', FileType='pdf', CropLevel=2)
Plot.PlotSolution(mesh, physics, solver, "Entropy", Equidistant=True, PlotExact=False, include_mesh=True, 
	ShowTriangulation=False, EqualAR=EqualAR)
Plot.SaveFigure(FileName='Entropy', FileType='pdf', CropLevel=2)
Post.get_boundary_info(mesh, physics, solver, "bottom", "Pressure", integrate=True, 
		vec=[1.,0.], dot_normal_with_vec=True, plot_vs_x=True, plot_vs_y=False, Label="F_x")
Plot.ShowPlot()

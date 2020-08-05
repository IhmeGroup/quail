import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import meshing.gmsh as MeshGmsh


def WriteMeshToText(mesh, FileName):
	# Unpack
	EG = mesh.ElemGroups[0]

	# Open file
	fo = open(FileName, "w")

	# Mesh dimension
	fo.write("%d %d %d\n" % (mesh.Dim, EG.QOrder, EG.nNodePerElem))

	# Nodes
	fo.write("$Nodes\n")
	fo.write("%d\n" % (mesh.nNode))
	for n in range(mesh.nNode):
		for d in range(mesh.Dim):
			fo.write("%.15E " % (mesh.Coords[n][d]))
		fo.write("\n")

	# Elements
	fo.write("$Elements\n")
	fo.write("%d\n" % (EG.nElem))
	for elem in range(EG.nElem):
		for n in range(EG.nNodePerElem):
			fo.write("%d " % (EG.Elem2Nodes[elem][n]))
		fo.write("\n")

	# Interior faces
	fo.write("$IFaces\n")
	fo.write("%d\n" % (mesh.nIFace))
	for IFace in mesh.IFaces:
		fo.write("%d %d %d %d\n" % (IFace.elemL_id, IFace.faceL_id,
			IFace.elemR_id, IFace.faceR_id))

	# Boundary groups
	fo.write("$BFaceGroups\n")
	fo.write("%d\n" % (mesh.nBFaceGroup))
	for BFG in mesh.BFaceGroups:
		fo.write("%s\n" % (BFG.Name))
		fo.write("%d\n" % (BFG.nBFace))
		for BFace in BFG.BFaces:
			fo.write("%d %d\n" % (BFace.elem_id, BFace.face_id))

	# Close file
	fo.close()



### Read mesh
MeshFile = "box_5x5.msh"
mesh = MeshGmsh.ReadGmshFile(MeshFile)
NewFile = "box_5x5.txt"
WriteMeshToText(mesh, NewFile)




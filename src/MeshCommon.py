from General import BasisType
from Mesh import *


def Mesh1D(Coords=None, nElem=10, Uniform=True, xmin=-1., xmax=1., Periodic=True):
	if Coords is None and not Uniform:
		raise Exception("Input error")

	### Create mesh
	if Coords is None:
		nNode = nElem + 1
		mesh = Mesh(dim=1, nNode=nNode)
		mesh.Coords = np.zeros([mesh.nNode,mesh.Dim])
		mesh.Coords[:,0] = np.linspace(xmin,xmax,mesh.nNode)
	else:
		Coords.shape = -1,1
		nNode = Coords.shape[0]
		nElem = nNode - 1
		mesh = Mesh(dim=1, nNode=nNode)
		mesh.Coords = Coords

	# IFaces
	if Periodic:
		mesh.nIFace = mesh.nNode - 1
		mesh.CreateIFaces()
		# mesh.IFaces = [IFace() for i in range(mesh.nIFace)]
		for i in range(mesh.nIFace):
			IFace_ = mesh.IFaces[i]
			IFace_.ElemL = i-1
			IFace_.faceL = 1
			IFace_.ElemR = i
			IFace_.faceR = 0
		# Leftmost face
		mesh.IFaces[0].ElemL = nElem - 1
	# Rightmost face
	# mesh.IFaces[-1].ElemR = 0
	else:
		mesh.nIFace = nElem - 1
		mesh.CreateIFaces()
		for i in range(mesh.nIFace):
			IFace_ = mesh.IFaces[i]
			IFace_.ElemL = i
			IFace_.faceL = 1
			IFace_.ElemR = i+1
			IFace_.faceR = 0
		# Boundary groups
		mesh.nBFaceGroup = 2
		mesh.CreateBFaceGroups()
		for i in range(mesh.nBFaceGroup):
			BFG = mesh.BFaceGroups[i]
			BFG.nBFace = 1
			BFG.CreateBFaces()
			BF = BFG.BFaces[0]
			BF.ElemGroup = 0
			if i == 0:
				BFG.Title = "Left"
				BF.Elem = 0
				BF.face = 0
			else:
				BFG.Title = "Right"
				BF.Elem = nElem - 1
				BF.face = 1
		


	# mesh.nBFaceGroup = 2
	# mesh.BFaceGroups = [BFaceGroups() for i in range(mesh.nBFaceGroup)]
	# Left

	# only one element group
	mesh.nElemGroup = 1
	# mesh.ElemGroups = [ElemGroup() for i in range(mesh.nElemGroup)]
	mesh.CreateElemGroups()
	EGroup = mesh.ElemGroups[0]
	EGroup.SetParams(QBasis=BasisType["SegLagrange"], QOrder=1, nElem=nElem)
	# EGroup.nFaceMax = np.amax(EGroup.nFace)
	# EGroup.Faces = [[Face()  for j in range(EGroup.nFacePerElem)] for i in range(EGroup.nElem)]
	EGroup.CreateFaces()
	# interior elements
	for elem in range(EGroup.nElem):
		for i in range(EGroup.nFacePerElem):
		# for i in range(EGroup.nFace[elem]):
			Face_ = EGroup.Faces[elem][i]
			Face_.Type = INTERIORFACE
			Face_.Number = elem + i

	# EGroup.Elem2Nodes = np.zeros([EGroup.nElem,EGroup.nNodePerElem], dtype=int)
	EGroup.CreateElem2Nodes()
	for elem in range(EGroup.nElem):
		for i in range(EGroup.nNodePerElem):
			EGroup.Elem2Nodes[elem][i] = elem + i

	mesh.Finalize()

	return mesh


def RefineUniform1D(Coords_old):
	nNode_old = len(Coords_old)
	nElem_old = nNode_old-1

	nElem = nElem_old*2
	nNode = nElem+1

	Coords = np.zeros([nNode,1])

	for n in range(nNode_old-1):
		Coords[2*n] = Coords_old[n]
		Coords[2*n+1] = np.mean(Coords_old[n:n+2])
	Coords[-1] = Coords_old[-1]

	return Coords











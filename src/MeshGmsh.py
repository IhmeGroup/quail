import numpy as np
import code
import Errors
import General
import Mesh
import Basis
import copy
import MeshTools


GMSHVERSION = "2.2"
NTYPES = 37


class GmshEntityInfo(object):
	def __init__(self):
		self.nNode = -1
		self.QOrder = -1
		self.QBasis = -1
		self.Shape = -1
		self.Supported = False
		self.NodeOrder = None


def CreateGmshEntitiesInfo():
	# (NTYPES+1) objects due to one indexing
	EntitiesInfo = [GmshEntityInfo() for n in range(NTYPES+1)]

	''' 
	Assume most element types are not supported
	Only fill in supported elements
	'''

	# Linear line segments
	EntityInfo = EntitiesInfo[1]
	EntityInfo.nNode = 2
	EntityInfo.QOrder = 1
	EntityInfo.QBasis = General.BasisType.SegLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 1])

	# Linear triangle
	EntityInfo = EntitiesInfo[2]
	EntityInfo.nNode = 3
	EntityInfo.QOrder = 1
	EntityInfo.QBasis = General.BasisType.TriLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 1, 2])

	# Linear quadrilateral
	EntityInfo = EntitiesInfo[3]
	EntityInfo.nNode = 4
	EntityInfo.QOrder = 1
	EntityInfo.QBasis = General.BasisType.QuadLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 1, 3, 2])

	# Quadratic line segment
	EntityInfo = EntitiesInfo[8]
	EntityInfo.nNode = 3
	EntityInfo.QOrder = 2
	EntityInfo.QBasis = General.BasisType.SegLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 2, 1])

	# Quadratic triangle
	EntityInfo = EntitiesInfo[9]
	EntityInfo.nNode = 6
	EntityInfo.QOrder = 2
	EntityInfo.QBasis = General.BasisType.TriLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 3, 1, 5, 4, 2])

	# Quadratic quadrilateral
	EntityInfo = EntitiesInfo[10]
	EntityInfo.nNode = 9
	EntityInfo.QOrder = 2
	EntityInfo.QBasis = General.BasisType.QuadLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 4, 1, 7, 8, 5, 3, 6, 2])

	# Point
	EntityInfo = EntitiesInfo[15]
	EntityInfo.nNode = 1
	EntityInfo.QOrder = 0
	EntityInfo.Shape = General.ShapeType.Point
	EntityInfo.Supported = True

	# Cubic triangle
	EntityInfo = EntitiesInfo[21]
	EntityInfo.nNode = 10
	EntityInfo.QOrder = 3
	EntityInfo.QBasis = General.BasisType.TriLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 3, 4, 1, 8, 9, 5, 7, 6, 2])

	# Quartic triangle
	EntityInfo = EntitiesInfo[23]
	EntityInfo.nNode = 15
	EntityInfo.QOrder = 4
	EntityInfo.QBasis = General.BasisType.TriLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 3, 4, 5, 1, 11, 12, 13, 6, 
									10, 14, 7, 9, 8, 2])

	# Cubic line segment
	EntityInfo = EntitiesInfo[26]
	EntityInfo.nNode = 4
	EntityInfo.QOrder = 3
	EntityInfo.QBasis = General.BasisType.SegLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 2, 3, 1])

	# Quartic line segment
	EntityInfo = EntitiesInfo[27]
	EntityInfo.nNode = 5
	EntityInfo.QOrder = 4
	EntityInfo.QBasis = General.BasisType.SegLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 2, 3, 4, 1])

	# Quintic line segment
	EntityInfo = EntitiesInfo[28]
	EntityInfo.nNode = 6
	EntityInfo.QOrder = 5
	EntityInfo.QBasis = General.BasisType.SegLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 2, 3, 4, 5, 1])

	# Cubic quadrilateral
	EntityInfo = EntitiesInfo[36]
	EntityInfo.nNode = 16
	EntityInfo.QOrder = 3
	EntityInfo.QBasis = General.BasisType.QuadLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 4, 5, 1, 11, 12, 13, 6, 10, 15, 14, 
									7, 3, 9, 8, 2])

	# Quartic quadrilateral
	EntityInfo = EntitiesInfo[37]
	EntityInfo.nNode = 25
	EntityInfo.QOrder = 4
	EntityInfo.QBasis = General.BasisType.QuadLagrange
	EntityInfo.Shape = Basis.Basis2Shape[EntityInfo.QBasis]
	EntityInfo.Supported = True
	EntityInfo.NodeOrder = np.array([0, 4, 5, 6, 1, 15, 16, 20, 17, 7,
								    14, 23, 24, 21, 8, 13, 19, 22, 18, 9,
								    3, 12, 11, 10, 2])

	return EntitiesInfo


class PhysicalGroup(object):
	def __init__(self):
		self.Dim = 1
		self.Group = -1
		self.Number = 0
		self.Name = ""


class FaceInfo(object):
	def __init__(self):
		self.nVisit = 1
		self.BFlag = 0
		self.Group = 0
		self.Elem = 0
		self.Face = 0
		self.nfnode = 0
		self.snodes = None
	def Set(self, **kwargs):
		for key in kwargs:
		    if hasattr(self, key):
		        setattr(self, key, kwargs[key])
		    else: 
		        raise AttributeError


def FindLineAfterString(fo, string):
	# Start from beginning
	fo.seek(0)

	# Compare line by line
	found = False
	while not found:
		fl = fo.readline()
		if not fl:
			# reached end of file
			break
		if fl.startswith(string):
			found = True

	if not found:
		raise Errors.FileReadError


def ReadMeshFormat(fo):
	# Find beginning of section
	FindLineAfterString(fo, "$MeshFormat")
	# Get Gmsh version
	fl = fo.readline()
	ver = fl.split()[0]
	if ver != GMSHVERSION:
		raise Exception("Unsupported version")
	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndMeshFormat"):
		raise Errors.FileReadError


def ReadPhysicalGroups(fo):
	# Find beginning of section
	FindLineAfterString(fo, "$PhysicalNames")
	# Number of physical names
	nPGroup = int(fo.readline())
	# Allocate
	PGroups = [PhysicalGroup() for i in range(nPGroup)]
	# Loop over entities
	for i in range(nPGroup):
		PGroup = PGroups[i]
		fl = fo.readline()
		ls = fl.split()
		PGroup.Dim = int(ls[0])
		PGroup.Number = int(ls[1])
		PGroup.Name = ls[2][1:-1]

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndPhysicalNames"):
		raise Errors.FileReadError

	return PGroups, nPGroup


def ReadNodes(fo, mesh):
	# Find beginning of section
	FindLineAfterString(fo, "$Nodes")
	# Number of nodes
	nNode = int(fo.readline())
	# Allocate nodes - assume 3D first
	Nodes = np.zeros([nNode,3])
	# Extract nodes
	for n in range(nNode):
		fl = fo.readline()
		ls = fl.split()
		# Explicitly use for loop for compatibility with
		# both Python 2 and Python 3
		for d in range(3):
			Nodes[n,d] = float(ls[d+1])
		# Sanity check
		if int(ls[0]) > nNode:
			raise Errors.FileReadError

	# Change dimension if needed
	ds = [0,1,2]
	for d in ds:
		# Find max perturbation from zero
		diff = np.amax(np.abs(Nodes[:,d]))
		if diff <= General.eps:
			# remove from ds
			ds.remove(d)

	# New dimension
	dim = len(ds)
	Nodes = Nodes[:,ds]

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndNodes"):
		raise Errors.FileReadError

	# Store in mesh
	mesh.Coords = Nodes
	mesh.nNode = nNode
	mesh.Dim = dim

	return mesh


def ReadMeshEntities(fo, mesh, PGroups, nPGroup, EntitiesInfo):
	# Find beginning of section
	FindLineAfterString(fo, "$Elements")
	# Number of entities (cells, faces, edges)
	nEntity = int(fo.readline())
	# Loop over entities
	for n in range(nEntity):
		fl = fo.readline()
		ls = fl.split()
		# Parse line
		enum = int(ls[0])
		etype = int(ls[1])
		PGnum = int(ls[3])
		
		found = False
		for PGidx in range(nPGroup):
			PGroup = PGroups[PGidx]
			if PGroup.Number == PGnum:
				found = True
				break
		if not found:
			raise Exception("Physical group not found!")

		if PGroup.Dim == mesh.Dim:
			### Entity is an element
			QOrder = EntitiesInfo[etype].QOrder
			QBasis = EntitiesInfo[etype].QBasis
			# Check for existing element group
			found = False
			for egrp in range(mesh.nElemGroup):
				EG = mesh.ElemGroups[egrp]
				if QOrder == EG.QOrder and QBasis == EG.QBasis:
					found = True
					break
			if found:
				EG.nElem += 1
			else:
				# Need new element group
				mesh.nElemGroup += 1
				mesh.ElemGroups.append(Mesh.ElemGroup(QBasis=QBasis,QOrder=QOrder))
		elif PGroup.Dim < mesh.Dim:
			### Boundary entity
			# Check for existing boundary face group
			found = False
			for ibfgrp in range(mesh.nBFaceGroup):
				BFG = mesh.BFaceGroups[ibfgrp]
				if BFG.Name == PGroup.Name:
					found = True
					break
			if not found:
				mesh.nBFaceGroup += 1
				BFG = Mesh.BFaceGroup()
				mesh.BFaceGroups.append(BFG)
				BFG.Name = PGroup.Name
				PGroup.Group = mesh.nBFaceGroup - 1
			BFG.nBFace += 1
		else:
			raise Exception("Mesh error")

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndElements"):
		raise Errors.FileReadError

	return mesh


def AddFaceToHash(Node2FaceHash, nfnode, nodes, BFlag, Group, Elem, Face):

	if nfnode <= 0:
		raise ValueError("Need nfnode > 1")

	snodes = np.zeros(nfnode, dtype=int)
	snodes[:] = nodes[:nfnode]

	# Sort nodes
	snodes = np.sort(snodes)

	# Check if face already exists in face hash
	Exists = False
	n0 = snodes[0]
	FaceInfos = Node2FaceHash[n0]
	for FInfo in FaceInfos:
		if np.array_equal(snodes, FInfo.snodes):
			Exists = True
			# increment number of visits
			FInfo.nVisit += 1
			break

	if not Exists:
		# If it doesn't exist, then add it
		FInfo = FaceInfo()
		FaceInfos.append(FInfo)
		FInfo.Set(BFlag=BFlag, Group=Group, Elem=Elem, Face=Face,
				nfnode=nfnode, snodes=snodes)

	return FInfo, Exists


def DeleteFaceFromHash(Node2FaceHash, nfnode, nodes):

	if nfnode <= 0:
		raise ValueError("Need nfnode > 1")

	snodes = np.zeros(nfnode, dtype=int)
	snodes[:] = nodes[:nfnode]

	# Sort nodes
	snodes = np.sort(snodes)

	# Check if face already exists in face hash
	n0 = snodes[0]
	FaceInfos = Node2FaceHash[n0]
	# if FaceInfos == []:
	# 	raise LookupError

	DelIdx = [] # for storing which indices to delete
	for i in range(len(FaceInfos)):
		FInfo = FaceInfos[i]
		found = False
		if np.array_equal(snodes, FInfo.snodes):
			found = True

		# If found, store for deletion later
		if found:
			DelIdx.append(i)

	# Delete
	for i in DelIdx:
		del FaceInfos[i]
		 

def FillMesh(fo, mesh, PGroups, nPGroup, EntitiesInfo):
	# Allocate additional mesh structures
	for ibfgrp in range(mesh.nBFaceGroup):
		BFG = mesh.BFaceGroups[ibfgrp]
		BFG.AllocBFaces()
	nFaceMax = 0
	for EG in mesh.ElemGroups:
		# also find maximum # faces per elem
		EG.AllocFaces()
		EG.AllocElem2Nodes()
		if nFaceMax < EG.nFacePerElem: nFaceMax = EG.nFacePerElem
	mesh.AllocHelpers() 

	# Over-allocate IFaces
	mesh.nIFace = mesh.nElemTot*nFaceMax
	mesh.AllocIFaces()

	# reset nIFace - use as a counter
	mesh.nIFace = 0

	# Dictionary for hashing
	# Node2FaceHash = {n:FaceInfo() for n in range(mesh.nNode)}
	Node2FaceHash = {n:[] for n in range(mesh.nNode)}

	# Go to entities section
	FindLineAfterString(fo, "$Elements")

	# Number of entities
	nEntity = int(fo.readline())
	bf = [0 for i in range(mesh.nBFaceGroup)] # BFace counter
	elem = 0 # elem counter
	# Loop through entities
	for e in range(nEntity):
		fl = fo.readline()
		ls = fl.split()
		# Parse line
		enum = int(ls[0])
		etype = int(ls[1])
		PGnum = int(ls[3])
		
		found = False
		for PGidx in range(nPGroup):
			PGroup = PGroups[PGidx]
			if PGroup.Number == PGnum:
				found = True
				break
		if not found:
			raise Exception("Physical group not found!")

		# Check if entity type is supported
		if not EntitiesInfo[etype].Supported:
			raise Exception("Entity type not supported")

		# Get nodes	
		nTag = int(ls[2]) # number of tags
			# see http://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_63.php
		offsetTag = 3 # 3 integers (including nTag) before tags start
		iStart = nTag + offsetTag # starting index of node numbering
		nn = EntitiesInfo[etype].nNode
		elist = ls[iStart:] # list of nodes (string format)
		if len(elist) != nn: 
			raise Exception("Wrong number of nodes")
		nodes = np.zeros(nn, dtype=int)
		for i in range(nn):
			# Convert to int one-by-one for compatibility with Python 2 and 3
			nodes[i] = int(elist[i]) - 1 # switch to zero index

		if PGroup.Group >= 0:
			### Boundary
			# Get basic info
			QBasis = EntitiesInfo[etype].QBasis
			QOrder = EntitiesInfo[etype].QOrder
			ibfgrp = PGroup.Group
			BFG = mesh.BFaceGroups[ibfgrp]
			# Number of q = 1 face nodes
			nfnode = Basis.Order2nNode(QBasis, 1) 
			# Add q = 1 nodes to hash table
			FInfo, Exists = AddFaceToHash(Node2FaceHash, nfnode, nodes, True, 
				ibfgrp, -1, bf[ibfgrp])
			bf[ibfgrp] += 1
		elif PGroup.Group == -1:
			### Interior element
			# Get basic info
			QOrder = EntitiesInfo[etype].QOrder
			QBasis = EntitiesInfo[etype].QBasis
			# Check for existing element group
			found = False
			for EG in mesh.ElemGroups:
				if QOrder == EG.QOrder and QBasis == EG.QBasis:
					found = True
					break
			# Sanity check
			if not found:
				raise Exception("Can't find element group")
			# Number of element nodes
			nnode = Basis.Order2nNode(QBasis, QOrder)
			# Sanity check
			if nnode != EntitiesInfo[etype].nNode:
				raise Exception("Check Gmsh entities")
			# Convert node ordering
			newnodes = nodes[EntitiesInfo[etype].NodeOrder]
			# Store in Elem2Nodes
			EG.Elem2Nodes[elem] = newnodes
			# Increment elem counter
			elem += 1
		else:
			raise ValueError

	# Verify footer
	fl = fo.readline()
	if not fl.startswith("$EndElements"):
		raise Errors.FileReadError

	# Fill boundary and interior face info
	for egrp in range(mesh.nElemGroup):
		EG = mesh.ElemGroups[egrp]
		for elem in range(EG.nElem):
			for face in range(EG.nFacePerElem):
				# Local q = 1 nodes on face
				fnodes, nfnode = Basis.LocalQ1FaceNodes(EG.QBasis, EG.QOrder, face)

				# Convert to global nodes
				fnodes[:] = EG.Elem2Nodes[elem][fnodes[:]]

				# Add to hash table
				FInfo, Exists = AddFaceToHash(Node2FaceHash, nfnode, fnodes, False, 
					egrp, elem, face)

				if Exists:
					# Face already exists in hash table
					if FInfo.nVisit != 2:
						raise Errors.FileReadError("More than two elements share a face " + 
							"or a boundary face is referenced by more than one element")

					# Link elem to BFace or IFace
					if FInfo.BFlag:
						# boundary face
						# Store in BFG
						BFG = mesh.BFaceGroups[FInfo.Group]
						# try:
						# 	BFace = BFG.BFaces[FInfo.Face]
						# except:
						# 	code.interact(local=locals())
						BFace = BFG.BFaces[FInfo.Face]
						BFace.ElemGroup = egrp; BFace.Elem = elem; BFace.face = face
						# Store in Face
						Face = EG.Faces[elem][face]
						Face.Group = FInfo.Group
						Face.Number = FInfo.Face
					else:
						# interior face
						# Store in IFace
						IFace = mesh.IFaces[mesh.nIFace]
						IFace.ElemGroupL = FInfo.Group
						IFace.ElemL = FInfo.Elem
						IFace.faceL = FInfo.Face
						IFace.ElemGroupR = egrp
						IFace.ElemR = elem
						IFace.faceR = face
						# Store in left Face
						Face = EG.Faces[FInfo.Elem][FInfo.Face]
						Face.Group = General.INTERIORFACE
						Face.Number = mesh.nIFace
						# Store in right face
						Face = EG.Faces[elem][face]
						Face.Group = General.INTERIORFACE
						Face.Number = mesh.nIFace
						# Increment IFace counter
						mesh.nIFace += 1

					DeleteFaceFromHash(Node2FaceHash, nfnode, fnodes)

	# Make sure no faces left in hash
	nleft = 0
	for n in range(mesh.nNode):
		FaceInfos = Node2FaceHash[n]
		for FInfo in FaceInfos:
			snodes = FInfo.snodes
			for k in range(FInfo.nfnode):
				print int(snodes[k]+1),
			print
			nleft += 1

	if nleft != 0:
		raise Errors.FileReadError("Mesh connectivity error: the above %d " % (nleft) +
			"face(s) remain(s) in the hash")

	# Resize IFace
	if mesh.nIFace > mesh.nElemTot*nFaceMax:
		raise ValueError
	mesh.IFaces = mesh.IFaces[:mesh.nIFace]

	mesh.FillFaces()

	# Check face orientations
	MeshTools.CheckFaceOrientations(mesh)



def ReadGmshFile(FileName):
	# Check file extension
	if FileName[-4:] != ".msh":
		raise Exception("Wrong file type")

	# Open file
	fo = open(FileName, "r")

	# Mesh object
	mesh = Mesh.Mesh(nElemGroup=0)

	# Object that stores Gmsh entity info
	EntitiesInfo = CreateGmshEntitiesInfo()

	# Read sections one-by-one
	ReadMeshFormat(fo)
	mesh = ReadNodes(fo, mesh)
	PGroups, nPGroup = ReadPhysicalGroups(fo)
	mesh = ReadMeshEntities(fo, mesh, PGroups, nPGroup, EntitiesInfo)

	# Create rest of mesh
	FillMesh(fo, mesh, PGroups, nPGroup, EntitiesInfo)

	# Print some stats
	print("%d elements in the mesh" % (mesh.nElemTot))
	
	# Done with file
	fo.close()

	return mesh

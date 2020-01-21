import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import MeshGmsh
import h5py


def WriteMeshToHDF5(mesh, FileName):
	# Unpack
	EG = mesh.ElemGroups[0]

	# write hdf5 file
	with h5py.File(FileName, 'w') as f:
		f.create_dataset('Dimension', data=mesh.Dim)
		f.create_dataset('QOrder', data=EG.QOrder)
		f.create_dataset('nNodePerElem', data=EG.nNodePerElem)
		f.create_dataset('nNode', data=mesh.nNode)
		f.create_dataset('NodeCoords', data=mesh.Coords)
		f.create_dataset('nElem', data=EG.nElem)
		f.create_dataset('Elem2Nodes', data=EG.Elem2Nodes)

		# Interior faces
		f.create_dataset('nIFace', data=mesh.nIFace)
		IFaceData = np.zeros([mesh.nIFace, 4], dtype=int)
		n = 0
		for IFace in mesh.IFaces:
			IFaceData[n,0] = IFace.ElemL
			IFaceData[n,1] = IFace.faceL
			IFaceData[n,2] = IFace.ElemR
			IFaceData[n,3] = IFace.faceR
			n += 1
		f.create_dataset('IFaceData', data=IFaceData)

		# Boundary groups
		f.create_dataset('nBFaceGroup', data=mesh.nBFaceGroup)
		for BFG in mesh.BFaceGroups:
			s = "BFG_" + BFG.Name + "_nBFace"
			f.create_dataset(s, data=BFG.nBFace)

			BFaceData = np.zeros([BFG.nBFace, 2], dtype=int)
			n = 0
			for BFace in BFG.BFaces:
				BFaceData[n,0] = BFace.Elem
				BFaceData[n,0] = BFace.face
				n += 1

			s = "BFG_" + BFG.Name + "_BFaceData"
			f.create_dataset(s, data=BFaceData)

		# Close
		f.close()



### Read mesh
MeshFile = "box_5x5.msh"
mesh = MeshGmsh.ReadGmshFile(MeshFile)
NewFile = "box_5x5.h5"
WriteMeshToHDF5(mesh, NewFile)

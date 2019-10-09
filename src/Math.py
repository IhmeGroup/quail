


def MatDetInv(A, d, detA, iA):
	if d == 1:
		det = A[0]
		if detA is not None: detA[0] = det;
		if iA is not None:
			if det == 0.:
				raise Exception("Singular matrix")
			iA[0] = 1./det
	elif d == 2:
		det = A[0,0]*A[1,1] - A[1,2]*A[2,1]
		if detA is not None: detA[0] = det;
		if iA is not None:
			if det == 0.:
				raise Exception("Singular matrix")
			iA[0] =  A[3]/det
			iA[1] = -A[1]/det
			iA[2] = -A[2]/det
			iA[3] =  A[0]/det;
	else:
		raise Exception("Can only deal with 2x2 matrices or smaller")


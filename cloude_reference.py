import numpy as np
from numpy.linalg import inv

def cloudeDecomposition(mm):
	M = np.copy(mm)
	# calculate coherency matrix H of Mueller Matrix M
	M[0][0] = 1.0
	
	H = np.reshape(np.zeros(16, dtype='complex64'), (4, 4))
	H[0][0] = M[0][0]    + M[1][1]    + M[2][2]    + M[3][3]
	H[0][1] = M[0][1]    + M[1][0]    - 1j*M[2][3] + 1j*M[3][2]
	H[0][2] = M[0][2]    + M[2][0]    + 1j*M[1][3] - 1j*M[3][1]  
	H[0][3] = M[0][3]    - 1j*M[1][2] + 1j*M[2][1] + M[3][0]  
	
	H[1][0] = M[0][1]    + M[1][0]    + 1j*M[2][3] - 1j*M[3][2]
	H[1][1] = M[0][0]    + M[1][1]    - M[2][2]    - M[3][3]
	H[1][2] = 1j*M[0][3] + M[1][2]    + M[2][1]    - 1j*M[3][0]
	H[1][3] = 1j*M[2][0] - 1j*M[0][2] + M[1][3]    + M[3][1]
	
	H[2][0] = M[0][2]    + M[2][0]    - 1j*M[1][3] + 1j*M[3][1]
	H[2][1] = M[1][2]    - 1j*M[0][3] + M[2][1]    + 1j*M[3][0]
	H[2][2] = M[0][0]    - M[1][1]    + M[2][2]    -  M[3][3]
	H[2][3] = 1j*M[0][1] - 1j*M[1][0] + M[2][3]    + M[3][2]
	
	H[3][0] = M[0][3]    + 1j*M[1][2] - 1j*M[2][1] + M[3][0]
	H[3][1] = 1j*M[0][2] - 1j*M[2][0] + M[1][3]    + M[3][1]
	H[3][2] = 1j*M[1][0] - 1j*M[0][1] + M[2][3]    + M[3][2]
	H[3][3] = M[0][0]    - M[1][1]    - M[2][2]    + M[3][3]
	
	H = np.divide(H, 4.0)
	
	eigValH, eigVecH = np.linalg.eig(H)
	
	# Eigenvectors of H must be real (H is hermitian).
	# Therefore we neglect the imaginary parts, since
	# they come from numerical uncertainties.
	# Real part of eigValH is in the range 1E-1 ... 1E-2
	# Imaginary part of eigValH is in the range 1E-29 ... 1E-31 
	eigValH = np.real(eigValH)
	eigVecH = np.real(eigVecH)
	
	# sort Eigenvalues and Eigenvectors descending by Eigenvalues
	idx = eigValH.argsort()[::-1]
	eigValH = eigValH[idx]
	eigVecH = eigVecH[:,idx]
	
	#------------------------------------------------------------------------
	# calculate Jones matrices (M_Ji, i=0...3) from the obtained Eigenvectors
	#------------------------------------------------------------------------
	
	# create empty jones and mueller matrices
	jonesMatrix = dict()
	cloudeDecom = dict()
	
	JtoMtransformer = np.array([[1.0, 0.0, 0.0,  1.0],
							    [1.0, 0.0, 0.0, -1.0],
							    [0.0, 1.0, 1.0,  0.0],
							    [0.0, 1j , -1j,  0.0]])
	invJtoMtransformer = np.array([[ 0.5,  0.5,  0.0,  0.0   ],
								   [ 0.0,  0.0,  0.5, -0.5*1j],
								   [ 0.0,  0.0,  0.5,  0.5*1j],
								   [ 0.5, -0.5,  0.0,  0.0   ]])
	
	for i in range(4):
		jonesMatrix[i] = np.reshape(np.zeros(4, dtype='complex64'), (2,2))
		cloudeDecom[i] = np.reshape(np.zeros(16, dtype='float32'), (4,4))
	
	for i in range(4):
		# assign jones matrix values from eigenvectors
		jonesMatrix[i][0][0] = eigVecH[i][0] + eigVecH[i][1]		# = r_pp
		jonesMatrix[i][0][1] = eigVecH[i][2] - 1j*eigVecH[i][3]		# = r_ps
		jonesMatrix[i][1][0] = eigVecH[i][2] + 1j*eigVecH[i][3]		# = r_sp
		jonesMatrix[i][1][1] = eigVecH[i][0] - eigVecH[i][1]		# = r_ss

		# make mueller matrix from jones matrix
		J = np.kron(jonesMatrix[i], np.conjugate(jonesMatrix[i]))
		J = np.dot(J, invJtoMtransformer)
		MJ = np.dot(JtoMtransformer, J)
		# Mueller matrix must be real, so remove the imaginary parts
		# since they are here always zero.
		cloudeDecom[i] = np.copy(np.real(MJ))
	
	# At this point we got the complete decomposited mueller matrix.
	# Now we delete all matrices which have a negative eigenvalue from
	# above as pre-factor and after that we divide each remaining matrix
	# by the sum of its eigenvalues.
	# Finally, summing up all remaining matrices delivers a Mueller matrix,
	# which is free from non-physical realizable parts and which is the
	# closest approximation to the original measured Mueller matrix.
	
	# This will be the final mueller matrix to be returned
	# with removed non-physical realizable part
	MM = np.reshape(np.zeros(16, dtype='float32'), (4,4))
	
	#for i in sorted(cloudeDecom.keys()):
		## divide matrix by the sum of its eigenvalues
		## if the decomposition factor obtained from H is > 0.0
		#if eigValH[i] > 0.0:
			##eValCD, eVecCD = np.linalg.eig(cloudeDecom[i])
			##eValCD = np.real(eValCD) # just to be sure it will be real ;-)
			##cloudeDecom[i] = np.multiply(cloudeDecom[i], (eigValH[i]/np.sum(eValCD)))
			#cloudeDecom[i] = eigValH[i] * cloudeDecom[i]
			#MM = np.add(MM, cloudeDecom[i])
			
	#eigValMM, eigVecMM = np.linalg.eig(MM)
	#MM /= np.sum(eigValMM)
	
	MM1 = eigValH[0]*cloudeDecom[0]
	MM2 = eigValH[1]*cloudeDecom[1]
	MM3 = eigValH[2]*cloudeDecom[2]
	MM4 = eigValH[3]*cloudeDecom[3]
	
	return MM1,MM2,MM3,MM4,eigValH

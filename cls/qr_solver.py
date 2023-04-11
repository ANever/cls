import copy
import numpy as np


def rotation_vec(x):
    e = np.zeros(len(x))
    e[0] = 1
    rot_vec_1 = x + np.sqrt(x.dot(x)) * e
    rot_vec_2 = x - np.sqrt(x.dot(x)) * e
    if rot_vec_1.dot(rot_vec_1) > rot_vec_2.dot(rot_vec_2):
        rot_vec = rot_vec_1
    else:
        rot_vec = rot_vec_2
        
    if np.sqrt(rot_vec.dot(rot_vec)) < 10**(-9):
        return rot_vec
    else:
        rot_vec = rot_vec / np.sqrt(rot_vec.dot(rot_vec))
    return rot_vec

def QR_housholder(A, b):
    R = copy.deepcopy(A)
    Q = np.eye(A.shape[0])
    u_0 = np.zeros(A.shape[0])
    r_side = copy.deepcopy(b)
    for j in range(min(A.shape)):
            u = rotation_vec(R[j:,j])
            if j==0:
                u_0 = u
            else:
                R[j:,j-1] = u
            R[j:,j:] = R[j:,j:] - 2*np.outer(u, u.dot(R[j:,j:]))
            Q[j:,j:] = Q[j:,j:] - 2*np.outer(u, u.dot(Q[j:,j:]))
            r_side[j:] = r_side[j:] - 2*u*(u.dot(r_side[j:]))
    return Q, R, r_side, u_0

def fast_dot_Q(right_side, u_0, R):
    right_side = right_side - 2*u_0*(u_0.dot(right_side))
    for j in range(1,min(R.shape)):
        u = R[j:,j-1]
        right_side[j:] = right_side[j:] - 2*u*(u.dot(right_side[j:]))
        result = right_side
    return result

def _solve(R, b):
    length = min(R.shape)
    x = np.zeros(length)
    for i in range(length-1, -1, -1):
        for j in range(length-1, -1, -1):
            if i < j:
                if abs(R[j,j]) < 1e-15:
                    pass
                    #print('vary small R[j,j]')
                    # b[i] = 0
                else:
                    b[i] -= b[j] * R[i,j] / R[j,j]
        if abs(R[i,i]) < 1e-13:
            #print('vary small R[j,j]')
            x[i] = 0
        else:
            x[i] = b[i] / R[i,i]
    return x

def QR_solve(A, b, method = 'gs'):
    # if method == 'householder':
    Q, R, b, u_0 = QR_housholder(A, b)
    # elif method == 'gs':

    # Q, R = qr_gs_modsr(A)
    # # print('doing gs\n', Q, R)
    # b = np.transpose(Q) @ b
    res = _solve(R, b)
    # res = np.linalg.solve(R, b)
    return res


def SVD_solve(A, b, threshold = 10**(-4), verbose = False):
    u, s ,vh = np.linalg.svd(A, full_matrices=True)
    
    cond = s[0]/s[-1]
    if (cond > 1e3) or verbose:
        print('Matrix conditioning:', cond)
    if verbose:
        print('SVD: depricated ', sum(s<threshold),'/',len(s),' singular values')
    s[s < threshold] = 0
    s[s > threshold] = 1/s[s > threshold]
    # pinv = np.transpose(vh) @ np.transpose(Sx) @ np.transpose(u)
    result = np.transpose(vh) @ ((np.transpose(u) @ b)[:len(s)]*s)
    return result

def qr_gs_modsr(A, type=np.float32):
    #is faster for grater n (number of variables), but slower for grater m (number of equations)
    A = np.array(A, dtype=type)
    (m,n) = np.shape(A) # Get matrix A's shape m - # of rows, m - # of columns
    Q = np.array(A, dtype=type)      # Q - matrix A
    R = np.zeros((n, n), dtype=type) # R - matrix of 0's    
    for k in range(n):
        for i in range(k):
            R[i,k] = np.transpose(Q[:,i]).dot(Q[:,k])
            Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]
        R[k,k] = np.linalg.norm(Q[:,k]); Q[:,k] = Q[:,k] / R[k,k]

    return -Q, -R  # Return the resultant negative matrices Q and R 


if (__name__ == '__main__'):
    
    A = np.array([[1,0,1],[0,1,1],[3,1,4], [1,1,4]])

    print('new qr decomp')
    print(qr_gs_modsr(A))

    #Примеры применения

    #стандартный

    b = np.array([1,1,0,1])

    x = QR_solve(A, b)

    print(x)

    x2 = QR_solve(A,b, method='householder')

    print(x2)
    print('difff', x-x2)

    #если есть много одинаковых матриц с разными правыми частями, можно считать быстрее

    b1 = np.array([1,1,0,1])
    b2 = np.array([1,2,4,1])
    b3 = np.array([1,6,5,1])

    Q, R, _, u_0 = QR_housholder(A, b)

    b1 = fast_dot_Q(b1, u_0, R)
    b2 = fast_dot_Q(b2, u_0, R)
    b3 = fast_dot_Q(b3, u_0, R)

    x1 = _solve(R, b1)
    x2 = _solve(R, b2)
    x3 = _solve(R, b3)
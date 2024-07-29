"""Load full simulation data of multi-dimenisonal systems. 
"""
import scipy.io as scio
from dso.task.pde.utils_v1 import *
# from dso.task.pde.utils_subgrid import *

def trapezoidal(x):
    """Trapezoidal weights for trapezoidal rule integration."""
    diff = np.diff(x, axis=0)
    weights = np.zeros(x.shape)
    weights[1:-1] = diff[1:] + diff[:-1]
    weights[0] = diff[0]
    weights[-1] = diff[-1]
    weights = weights / 2
    return weights

def load_data(dataset,noise_level=0, data_amount = 1, training=False, cut_ratio =0.03):
    """
    load data and pass them to the corresponding PDE task 
    """
    X = []
    
    # noise_path = f'./dso/task/pde/noise_data_new/{dataset}_noise={noise_level}_data_ratio={data_amount}.npz'
    n_state_var = 1
    if dataset == 'advection':
        data = scio.loadmat('./dso/task/pde/data/advection.mat')
        ExactGreen = data.get('ExactGreen')[0]
        F = data.get('F')
        u_hom = data.get('U_hom')
        u = []
        xg = data.get('X')
        yg = data.get('Y')
        weight_y = trapezoidal(yg)
        XG, YG = np.meshgrid(xg, yg)
        ut = data.get('U')
        u.append(XG)
        u.append(YG)
        X.append(F)
        X.append(xg)
        X.append(yg)
        X.append(weight_y)
        X.append(u_hom)

        t = []
        sym_true = 'add,mul,mul,const,sub,mul,u1,u2,u1,exp,mul,const,sub,u1,u2,mul,mul,const,exp,mul,const,sub,u1,u2,sub,mul,u1,u2,u2'
        # sym_true = 'mul,mul,const,exp,mul,const,sub,u1,u2,add,mul,sub,mul,u1,u2,u1,less,u1,u2,mul,sub,mul,u1,u2,u2,large,u1,u2'
        n_input_var = 0

    elif dataset == 'helmholtz':
        data = scio.loadmat('./dso/task/pde/data/helmholtz.mat')
        ExactGreen = data.get('ExactGreen')[0]
        F = data.get('F')
        u_hom = data.get('U_hom')
        u = []
        xg = data.get('X')
        yg = data.get('Y')
        weight_y = trapezoidal(yg)
        XG, YG = np.meshgrid(xg, yg)
        ut = data.get('U')
        u.append(XG)
        u.append(YG)
        X.append(F)
        X.append(xg)
        X.append(yg)
        X.append(weight_y)
        X.append(u_hom)

        t = []
        sym_true = 'mul,mul,sin,mul,sub,const,u1,const,add,u2,add,const,mul,const,u2,u2'
        n_input_var = 0

    elif dataset == 'negative_helmholtz':
        data = scio.loadmat(f'./dso/task/pde/data/{dataset}.mat')
        ExactGreen = data.get('ExactGreen')[0]
        F = data.get('F')
        u_hom = data.get('U_hom')
        u = []
        xg = data.get('X')
        yg = data.get('Y')
        weight_y = trapezoidal(yg)
        XG, YG = np.meshgrid(xg, yg)
        ut = data.get('U')
        u.append(XG)
        u.append(YG)
        X.append(F)
        X.append(xg)
        X.append(yg)
        X.append(weight_y)
        X.append(u_hom)

        t = []
        sym_true = 'mul,const,mul,sinh,mul,const,u1,sinh,mul,const,sub,u2,const'
        n_input_var = 0

    elif dataset == 'laplace':
        data = scio.loadmat('./dso/task/pde/data/laplace.mat')
        ExactGreen = data.get('ExactGreen')[0]
        F = data.get('F')
        u = []
        xg = data.get('X')
        yg = data.get('Y')
        u_hom = data.get('U_hom')
        weight_y = trapezoidal(yg)
        XG, YG = np.meshgrid(xg, yg)
        ut = data.get('U')
        u.append(XG)
        u.append(YG)
        X.append(F)
        X.append(xg)
        X.append(yg)
        X.append(weight_y)
        X.append(u_hom)
        t = []
        # sym_true = 'add,sub,u1,mul,u1,u2,sub,u2,mul,u1,u2'
        sym_true = 'add,add,u1,mul,u1,u2,add,u2,mul,u1,u2'
        n_input_var = 0

    # elif dataset == 'schrodinger':
    elif dataset == 'potential_barrier':
        data = scio.loadmat(f'./dso/task/pde/data/{dataset}.mat')
        u = []
        x_G = data.get('XG').flatten()[:, None].astype(dtype='float32')
        y_G = data.get('YG').flatten()[:, None].astype(dtype='float32')
        X_G, Y_G = np.meshgrid(x_G, y_G)
        ut = np.genfromtxt(f'./dso/task/pde/data/Green_{dataset}_rational_0.csv',delimiter=',')
        u.append(X_G)
        u.append(Y_G)
        t = []
        # sym_true = 'add,sub,u1,mul,u1,u2,sub,u2,mul,u1,u2'
        sym_true = 'add,mul,sin,u1,sin,u2,mul,sin,u2,sin,u1'
        n_input_var = 0

    elif dataset == 'periodic_helmholtz':
        data = scio.loadmat(f'./dso/task/pde/data/{dataset}.mat')
        F = data.get('F')
        u = []
        xg = data.get('X')
        yg = data.get('Y')
        u_hom = data.get('U_hom')
        weight_y = trapezoidal(yg)
        XG, YG = np.meshgrid(xg, yg)
        ut = data.get('U')
        u.append(XG)
        u.append(YG)
        X.append(F)
        X.append(xg)
        X.append(yg)
        X.append(weight_y)
        X.append(u_hom)
        t = []
        sym_true = 'mul,const,sin,mul,const,add,const,sub,u1,u2'
        n_input_var = 0

    else:
        # assert False, "Unknown dataset"
        data = scio.loadmat(f'./dso/task/pde/data/{dataset}.mat')
        F = data.get('F')
        u = []
        xg = data.get('X')
        yg = data.get('Y')
        u_hom = data.get('U_hom')
        weight_y = trapezoidal(yg)
        XG, YG = np.meshgrid(xg, yg)
        ut = data.get('U')
        u.append(XG)
        u.append(YG)
        X.append(F)
        X.append(xg)
        X.append(yg)
        X.append(weight_y)
        X.append(u_hom)
        t = []
        sym_true = 'sub,u1,u2'
        n_input_var = 0

    return u, X, t, ut, sym_true, n_input_var, None, n_state_var


if  __name__ ==  "__main__":
    import time
    st = time.time()
    u = np.random.rand(500,200)
    x = np.random.rand(500,1)
    su = np.sum(Diff3(u,x,0))


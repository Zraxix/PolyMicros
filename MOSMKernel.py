import torch
from torch.func import vmap
from scipy.stats import qmc

def MultiouputSpectralMixtureKernel(r,i,j,params, valid_check = True):
    '''
    MultiouputSpectralMixtureKernel
    Written by Michael Buzzy

    Ref: https://arxiv.org/pdf/1709.01298

    Description: A general parametric kernel which can be used to construct compatable 
    auto- and cross-correlations. Package requires torch and all arrays to be torch arrays

    args:
        r: Array of position vectors. Shape: (num_positions, num_dimensions)
        i: Task/Phase identifier #1, int
        j: Task/Phase identifier #2, int
        valid_check: If true runs several tests to ensure parameters 
                     are formatted correctly and in the proper range. 

        params: Dictonary of parameters: 
            cov: Diagonal of covariance matrix (n_mixtures, n_tasks, n_dimensions)
            u  : Means of mixture elements     (n_mixtures, n_tasks, n_dimensions)
            w  : Mixture Weights               (n_mixtures, n_tasks)
            d  : Cross-Corr Spatial Delay      (n_mixtures, n_tasks, n_dimensions)
            p  : Cross-Corr phase shifts       (n_mixtures, n_tasks)

    returns:
        k  : kernel evaluations  (n_positions, )

    '''
    _, n_dimensions = r.shape
    n_mixtures, n_tasks, _ = params["cov"].shape

    if valid_check:
        assert n_dimensions == params["cov"].shape[2] == params["u"].shape[2] == params["d"].shape[2], "n_dimensions do not match across all params and r"
        assert n_mixtures == params["cov"].shape[0] == params["u"].shape[0] == params["d"].shape[0]== params["w"].shape[0]== params["p"].shape[0], "n_mixtures do not match across all params"
        assert n_tasks == params["cov"].shape[1] == params["u"].shape[1] == params["d"].shape[1]== params["w"].shape[1]== params["p"].shape[1], "n_tasks do not match across all params"

    #Construct Vectorized operations to batch over n_mixtures
    b_inv, b_matmul, b_dot, b_det = vmap(torch.linalg.inv), vmap(torch.matmul), vmap(torch.dot), vmap(torch.linalg.det)
    b_exp, b_cos, b_diag  = vmap(torch.exp), vmap(torch.cos), vmap(torch.diag)

    #extract needed params
    cov_i, cov_j = b_diag(params["cov"][:,i,:])**2, b_diag(params["cov"][:,j,:])**2 #these are vec's need to be mats duh!

    u_i, u_j = params["u"][:,i,:], params["u"][:,j,:]
    w_i, w_j = params["w"][:,i], params["w"][:,j]
    d_i, d_j = params["d"][:,i,:], params["d"][:,j,:]
    p_i, p_j = params["p"][:,i], params["p"][:,j]

    #Compute common quantites
    cov_sum_inv = b_inv(cov_i+cov_j)
    mean_diff_vec = u_i-u_j

    #Construct variables
    cov = 2*b_matmul(
        b_matmul(cov_i,cov_sum_inv),
        cov_j
        )
    
    u = b_matmul(
        cov_sum_inv,
        b_matmul(cov_i,u_j) + b_matmul(cov_j,u_i)
    )

    w = w_i*w_j*b_exp( 
        (-1/4)*b_dot(
            mean_diff_vec,
            b_matmul(cov_sum_inv,mean_diff_vec)
        )
    )

    a = w * (2*torch.pi)**(n_dimensions/2) * b_det(cov)**(1/2)

    d = d_i-d_j
    p = p_i-p_j

    if valid_check:
        assert cov.shape == (n_mixtures,n_dimensions,n_dimensions)
        assert u.shape == (n_mixtures,n_dimensions)
        assert d.shape == (n_mixtures,n_dimensions)

    #Evaluate Kernel
    k = lambda r:  torch.sum(
        a * b_exp(
        (-1/2)*b_dot(
            r+d,
            b_matmul(cov,r+d)
        )
        ) * b_cos(
            b_dot(r+d, u) + p
        )
    )
    
    b_k = vmap(k)

    return b_k(r)
    
def LHSParams(n_samples, n_tasks, n_mix, n_dim, bounds_dict, seed = 1234, dev = "cpu"):
    s1 = n_mix*n_tasks*n_dim
    s2 = n_mix*n_tasks
    n_params = 3*s1 + 2*s2

    l_bounds = [bounds_dict["cov"][0]]*s1 + [bounds_dict["u"][0]]*s1 + [bounds_dict["d"][0]]*s1 + [bounds_dict["w"][0]]*s2 + [bounds_dict["p"][0]]*s2 
    u_bounds = [bounds_dict["cov"][1]]*s1 + [bounds_dict["u"][1]]*s1 + [bounds_dict["d"][1]]*s1 + [bounds_dict["w"][1]]*s2 + [bounds_dict["p"][1]]*s2

    print(f"[INFO] MOSMKernel has {n_params} parameters.")

    sampler = qmc.LatinHypercube(n_params,seed = seed)
    samples = sampler.random(n=n_samples)
    samples_scaled = qmc.scale(samples, l_bounds, u_bounds)

    param_dict_list = [
        {
            "cov":torch.tensor(samples_scaled[i,:s1].reshape(n_mix,n_tasks,n_dim)).to(dev),
            "u":  torch.tensor(samples_scaled[i,s1:2*s1].reshape(n_mix,n_tasks,n_dim)).to(dev),
            "d":  torch.tensor(samples_scaled[i,2*s1:3*s1].reshape(n_mix,n_tasks,n_dim)).to(dev),
            "w":  torch.tensor(samples_scaled[i,3*s1:3*s1+s2].reshape(n_mix,n_tasks)).to(dev),
            "p":  torch.tensor(samples_scaled[i,3*s1+s2:3*s1+2*s2].reshape(n_mix,n_tasks)).to(dev),
            }

            for i in range(n_samples)
    ]

    return param_dict_list

if __name__ == "__main__":
    print("File run as main. Not Imported. Running Tests!")

    n_tasks, n_mix, n_dim, edge = 3,4,2,9

    params = {"cov":3*torch.ones((n_tasks,n_mix,n_dim)),
              "u":torch.rand((n_tasks,n_mix,n_dim)),
              "w":.03*torch.rand((n_tasks,n_mix)),
              "d":torch.rand((n_tasks,n_mix,n_dim))-.5,
              "p":2*torch.pi*torch.rand((n_tasks,n_mix))
              }
    
    #set up mesh of positions
    size = (edge,edge)
    xi = torch.linspace(-1*torch.pi,torch.pi,edge)
    yi = torch.linspace(-1*torch.pi,torch.pi,edge)
    pos = torch.stack(torch.meshgrid(xi,yi),axis=-1)
    pos = pos.reshape(-1,n_dim)
    points = pos.shape[0]

   
    stats = torch.stack([torch.fft.ifftshift(MultiouputSpectralMixtureKernel(pos,0,j,params).reshape(size)) for j in range(n_tasks)],-1)
    
    cov = torch.zeros((points*n_tasks,points*n_tasks))

    for a in range(n_tasks):
        for b in range(n_tasks):
            for i in range(points):
                for j in range(points):
                    cov[i+(a*points),j+(b*points)] = MultiouputSpectralMixtureKernel(pos[i].reshape(1,-1)-pos[j].reshape(1,-1),a,b,params)[0]

    print(f"Covariance Shape: {cov.shape}")
    print("\n")
    print("Symmetric Autocorrs:")

    for i in range(n_tasks):
        print(torch.all(cov[i*points:(i+1)*points,i*points:(i+1)*points]==cov[i*points:(i+1)*points,i*points:(i+1)*points].T))
    print("\n")

    print("Positive Semi-Definite:")
    eigvals = torch.linalg.eigvals(cov).real
    print(torch.all(eigvals > 0))
    print(torch.sum(eigvals < 0))
    print(torch.min(eigvals))
    print("\n")

    print("PSD Autocorrs:")
    for i in range(n_tasks):
        print(torch.all(torch.linalg.eigvals(cov[i*points:(i+1)*points,i*points:(i+1)*points]).real > 0))
    print("\n")

    print("Autocorrelaion Positive Spectrum:")
    print(torch.all(torch.fft.fftn(stats[...,0]).real>0))
    print(torch.min(torch.fft.fftn(stats[...,0]).real))
    print("\n")

    print("Autocorrelaion Zero Point Maximum:")
    print(stats[...,0][0,0]==torch.max(stats[...,0]))
    print("\n")

    print("K_ii(r) = K_ii(-r):")
    for i in range(n_tasks):
        print(torch.all(MultiouputSpectralMixtureKernel(pos,i,i,params) == MultiouputSpectralMixtureKernel(-1*pos,i,i,params)))


######### Plots
    import numpy as np
    import matplotlib.pyplot as plt
    stats = stats.detach().numpy()
    fig, ax = plt.subplots(2, 3, figsize=(12, 4))  # Optionally, adjust figure size

    # First subplot
    im0 = ax[0][0].imshow(np.fft.fftshift(stats[..., 0]))
    ax[1][0].imshow(stats[..., 0])
    ax[0][0].set_title("0-0 Corr")
    ax[0][0].set_axis_off()
    ax[1][0].set_axis_off()
    plt.colorbar(im0, ax=ax[0][0],shrink=.9)

    # Second subplot
    im1 = ax[0][1].imshow(np.fft.fftshift(stats[..., 1]))
    ax[1][1].imshow(stats[..., 1])
    ax[0][1].set_title("0-1 Corr")
    ax[0][1].set_axis_off()
    ax[1][1].set_axis_off()
    plt.colorbar(im1, ax=ax[0][1],shrink=.9)

    # Third subplot
    im2 = ax[0][2].imshow(np.fft.fftshift(stats[..., 2]))
    ax[1][2].imshow(stats[..., 2])
    ax[0][2].set_title("0-2 Corr")
    ax[0][2].set_axis_off()
    ax[1][2].set_axis_off()
    plt.colorbar(im2, ax=ax[0][2],shrink=.9)
    
    plt.tight_layout()
    # Save the figure
    plt.savefig("corr.png")
    plt.close()

    
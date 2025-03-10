import torch
import h5py
from einops import rearrange, reduce
from rich.progress import track
import numpy as np

import json
import argparse

from diffusion import Diffusion

from Unet import UnetND
from CUBEPlot import plot_cube, plot_grid
from ROGSHUtils import ROGSH432
GSH = ROGSH432()

from functools import partial

from MOSMKernel import MultiouputSpectralMixtureKernel, LHSParams

from GRFSampler import TwoPCorrelation
from GRFSampler import StatisticsGenerator


import matplotlib.pyplot as plt


import random

#load files for viz

CSRES = h5py.File("CSResults.h5","r")
PM3D = h5py.File("PM3D.h5","r")
PM3DO = h5py.File("PM3DO.h5","r")
TS = h5py.File("data/Test_Structures.h5","r")
PATCH = h5py.File("data/poly_patched.h5","r")


#Exemplar structures from dataset

'''
idx = random.choices(list(PM3D["poly"].keys()),k=6)
structs = []
for i in idx:
    structs.append(PM3D[f"poly/{i}"][:])

structs = np.stack(structs,axis=0)
plot_grid(structs*.5 + .5, f"figs/DGEN/DGEN_SMPL_SP.png",nrow=2)


for l in range(3):
    idx = random.choices(list(PM3D["poly"].keys()),k=12)
    structs = []
    for i in idx:
        structs.append(PM3D[f"poly/{i}"][:])

    structs = np.stack(structs,axis=0)
    #structs = rearrange(structs, "b c x y z -> b x y z c")
    plot_grid(structs*.5 + .5, f"figs/DGEN/DGEN_SMPL_FP_{l}.png",nrow=4)


idx = random.choices(list(PM3D["poly"].keys()),k=24)
structs = []
for i in idx:
     

'''

#Exemplar Neighboorhoods from each of the 5 experimental sources. 
'''
hoods = list(PATCH.keys())
del hoods[1]
print(hoods)


for hood in hoods:
    structs = PATCH[hood][:]
    idx = random.choices(list(range(structs.shape[0])),k=6)
    structs = structs[idx]
    structs = rearrange(structs,"b c x y z -> b x y z c")
    plot_grid(structs*.5 + .5, f"figs/LocalRef/{hood}_SMPL.png",nrow=2)


structs = []
for hood in hoods:
    neigh = PATCH[hood][:]
    idx = random.choice(list(range(neigh.shape[0])))

    structs.append(neigh[idx])

structs = np.stack(structs,axis=0)
structs = rearrange(structs,"b c x y z -> b x y z c")
print(structs.shape)

labels = ["A)","B)","C)","D)","E)"]

for i in range(len(hoods)):
    plot_cube(structs[i]*.5 + .5, f"figs/LocalRef/HOODS_{i}.png")
'''
#MOSM  KERNEL PLOTS

'''
dev = "cpu"
n_samples, n_mix, n_tasks, n_dim, edge = 12, 4, 3, 3, 128

bounds_dict = {"cov":[1.5,5],
              "u":[-5,5],
              "w":[-.02,.02],
              "d":[-0.5,0.5],
              "p":[0,2*torch.pi],
              }

params_dict = LHSParams(n_samples, n_tasks, n_mix, n_dim, bounds_dict, dev = dev)

#set up mesh of positions
ratio = (1,1,1)
size = (edge*ratio[0],edge*ratio[1],edge*ratio[2])
xi = torch.linspace(-1*torch.pi*ratio[0],torch.pi*ratio[0],size[0])
yi = torch.linspace(-1*torch.pi*ratio[1],torch.pi*ratio[1],size[1])
zi = torch.linspace(-1*torch.pi*ratio[2],torch.pi*ratio[2],size[2])
pos = torch.stack(torch.meshgrid(xi,yi,zi,indexing="ij"),axis=-1)
pos = torch.fft.fftshift(pos,dim=tuple(range(0,n_dim)))
pos = pos.reshape(-1,n_dim).to(dev)
points = pos.shape[0]


z = 0

stats_lst = []
grf   = []
for s in track(range(n_samples)):
    stats= torch.stack([MultiouputSpectralMixtureKernel(pos,0,j,params_dict[s]).reshape(size)  for j in range(n_tasks)],-1)
    stats_lst.append(stats)
    grf.append(StatisticsGenerator(stats, zmean = True))


structs = torch.stack(grf,dim=0)

for i in range(structs.shape[0]):
    plot_cube(structs[i].clip(-1,1)*.5 + .5, f"figs/GlobalApprox/GRF_Samples_{i}.png",)



#plot_grid(structs.clip(-1,1)*.5 + .5, f"figs/GlobalApprox/GRF_Samples.png",nrow=4)
exit()
pstats = stats_lst #rearrange(stats_lst[0],"x y z c -> c x y z")

for i in range(len(pstats)):
    plot_cube(torch.fft.fftshift(pstats[i][...,0]), f"figs/GlobalApprox/synth_stats_{i}.png", mode = "scalar",
                kind = "interior",
                #cbar = True,
                elev=20,
                azim=-40)

'''
#Test Structure  PLOTS
'''
structs = [TS["CS1"][:], TS["CS2"][:]]

structs = np.stack(structs,axis=0)
print(structs.shape)

labels = ["A)","B)"]

plot_grid(structs*.5 + .5, f"figs/CS1/Test_structs.png",nrow=1,label=True, labels=labels)
'''
#CS1 Plots

'''
target = TS["CS1"][:]
print(target.shape)
plot_cube(target*.5 + .5, f"figs/CS1/target.png")

structs = CSRES["CS1/TS1_EFS"][:]

plt.imshow(np.rot90(target[20:52,-32:,-1]*.5 + .5,k=1,axes=(0, 1)))
plt.axis("off")
plt.tight_layout()
plt.savefig(f"figs/CS1/2DPatch.png")

for i in range(10):
    plt.imshow(np.rot90(structs[i][20:52,-32:,-1]*.5 + .5,k=1,axes=(0, 1)))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"figs/CS1/2DPatch_{i}.png")

    plt.close()


target_stats = TwoPCorrelation(torch.tensor(target)).numpy()
'''
'''
EFS = np.zeros(target.shape)
EFS.fill(None)
for i in range(target.shape[2]//4):
    EFS[-i,:,:,:] = target[-4*i,:,:,:]

plot_cube(EFS*.5 + .5, f"figs/CS1/LowRes.png")
'''


'''
struct_stats = []
for struct in structs:
    struct_stats.append(
        TwoPCorrelation(torch.tensor(target)).numpy()
    )

struct_stats= np.stack(struct_stats,axis=0)

print(struct_stats.shape)

err = reduce(
    100*np.abs(
        (target_stats[None]-struct_stats)/target_stats[None]
        )
    ,"b x y z c -> x y z c", "mean")

print(np.mean(err))

exit()
plot_cube(
        np.fft.fftshift(err[...,0]),
        f"figs/CS1/EFS_ERR.png",
        mode = "scalar",
        kind = "interior",
        cbar = True,
        elev=20,
        azim=-40,
        figsize=(6,4)
        )


var = reduce(structs, "b x y z c -> x y z c",np.var)
plot_cube(var[...,0], f"figs/CS1/EFS_VAR.png", mode = "scalar",cbar = True,figsize=(6,4),cmap="binary")
'''

'''
for i in range(10):
    plot_cube(structs[i]*.5 + .5, f"figs/CS1/EFS_Samples_{i}.png")
'''

'''
#var = reduce(structs, "b x y z c -> x y z c",np.var)
err = reduce(
    np.abs(
        (target[None]-structs)/target[None]
        )
    ,"b x y z c -> x y z", "mean")


plot_cube(err.clip(0,25), f"figs/CS1/EFS_ERR.png",mode = "scalar",cbar = True,figsize=(6,4),cmap="inferno")


import matplotlib.pyplot as plt
import seaborn as sns

sns.ecdfplot(data=err.flatten())
plt.xlabel("Pointwise RMSE") 
plt.xlim([0,.6])
plt.savefig("figs/CS1/cdf.png",dpi=600)
plt.close()


plot_cube(err, f"figs/CS1/EFS_ERR.png", mode = "scalar",cbar = True,figsize=(6,4),cmap="inferno")

plot_cube(var[...,0], f"figs/CS1/EFS_VAR.png", mode = "scalar",cbar = True,figsize=(6,4),cmap="cubehelix")
'''
#CS2 Plots
target = TS["CS1"][:]

plot_cube(target*.5 + .5, f"figs/CS2-2/T12_Target.png")

print(target.shape)
target_stats = TwoPCorrelation(torch.tensor(target))
print(target_stats.shape)



'''
blank = np.ones_like(target)
plot_cube(blank, f"figs/CS2/blank.png")


import matplotlib.pyplot as plt

plt.imshow(target[:,:,0,:]*.5+.5)
plt.axis('off')
plt.savefig("figs/CS2/2d.png")
plt.close()
'''

structs = CSRES["CS2/TS1"][:]
structs = rearrange(structs,"b c x y z -> b x y z c")

'''
for i in [0,-1,-2,1,8,11]:
    plot_cube(structs[i]*.5 + .5, f"figs/CS2/TS2_Samples_{i}.png")


plot_cube(
        np.roll(torch.fft.fftshift(target_stats[...,0]).detach().cpu().numpy(),64,axis=1),
        f"figs/CS2/Target_Stats_{0}_tmp.png",
        mode = "scalar",
        )

'''

stats = []
for i in range(structs.shape[0]):
    stats.append(TwoPCorrelation(torch.tensor(structs[i])))

stats = torch.stack(stats,dim=0)
var = reduce(stats, "b x y z c -> x y z c",torch.var)
mean = reduce(stats, "b x y z c -> x y z c","mean")
err = reduce((target_stats[None]-stats)**2,"b x y z c -> x y z c", "mean")

plot_grid(structs[[0,1,2,3,4,5]]*.5 + .5, f"figs/CS2-2/TS2_Samples.png",nrow=2) #structs[[0,-1,-2,1,8,11]]

bounds =  [[None,None],[None,None],[None,None]]#[[0,.05],[-.01,.03],[-.01,.015]]
bounds_err = [[None,None],[None,None],[None,None]]# [[0,.8*1e-4],[0,.05*1e-4],[0,.05*1e-4]]


for corr in range(3):
    plot_cube(
        torch.fft.fftshift(target_stats[...,corr]).detach().cpu().numpy(),
        f"figs/CS2-2/Target_Stats_{corr}.png",
        mode = "scalar",
        kind = "interior",
        cbar = True,
        elev=20,
        azim=-40,
        figsize=(6,4),
        vmin=bounds[corr][0],
        vmax=bounds[corr][1]
        )
    
    plot_cube(
        torch.fft.fftshift(mean[...,corr]).detach().cpu().numpy(),
        f"figs/CS2-2/Target_Mean_{corr}.png",
        mode = "scalar",
        kind = "interior",
        cbar = True,
        elev=20,
        azim=-40,
        figsize=(6,4),
        vmin=bounds[corr][0],
        vmax=bounds[corr][1]
        )
    
    plot_cube(
        torch.fft.fftshift(err[...,corr]).detach().cpu().numpy(),
        f"figs/CS2-2/Target_Err_{corr}.png",
        mode = "scalar",
        kind = "interior",
        cbar = True,
        elev=20,
        azim=-40,
        figsize=(6,4),
        cmap="jet",
        vmin=bounds_err[corr][0],
        vmax=bounds_err[corr][1]
        )
    
    plot_cube(
        torch.fft.fftshift(var[...,corr]).detach().cpu().numpy(),
        f"figs/CS2-2/Target_var_{corr}.png",
        mode = "scalar",
        kind = "interior",
        cbar = True,
        elev=20,
        azim=-40,
        figsize=(6,4),

        )

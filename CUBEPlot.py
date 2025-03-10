'''
CubePlot: Volumetric Plotting tool in matplotlib
Written By:  Buzzy
Additions By: Conlain

"Because the voxels method in matplotlib is too damn slow" - Buzzy Circa 2024

Only works for volumes of equal edge edge lengths (XYZ) for now.

Warning: Theres a lottttttttttt of spinning arrays around in this....

THIS STILL NEEDS WORK FOR SURE.

NEEDS Documentation

NEEDS Not to only work for cubes

NEEDS to no be slow as a turtle
'''

import numpy as np
import matplotlib.pyplot as plt
from  PIL import Image
from einops import rearrange
import io

import torch

def plot_cube(
          im,
          savedir = None,
          mode = "rgb", #scalar or rgb,
          kind  = "surface", #surface: surface of volume, interior: for 3 orthogonal slices through the center
          elev = 30,
          azim=-45, 
          ortho = True, 
          edges = True, 
          coord = False, 
          cmap  = "viridis", 
          cbar  = False,
          title = None,
          vmin = None,
          vmax = None,
          edges_kw =  dict(color='k', linewidth=.5,linestyle="--", zorder=1e3),
          figsize = (1.9,1.9)
          ):
        
        #rotate the thing to the coord sys we want to visualize in
        im = np.flip(np.rot90(np.rot90(im, k=1, axes=(0, 1)), k=-1, axes=(1, 2)), axis=1)

        if mode == "scalar":

            if vmin is None:
                vmin = im.min()
            if vmax is None:
                vmax = im.max()

            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            im = plt.get_cmap(cmap)(norm(im))

        if kind == "interior":
            Cx = im[im.shape[0]//2,:,:]
            Cy = im[:,im.shape[0]//2,:]
            Cz = im[:,:,im.shape[0]//2]
        else:
            Cx = im[0,:,:]
            Cy = im[:,0,:]
            Cz = im[:,:,0]

        fig = plt.figure(figsize=figsize,dpi=600)
        ax = fig.add_subplot(111, projection='3d')
        ax.dist=6.2
        ax.view_init(elev=elev, azim=azim)
        ax.axis("off")

        if kind == "interior":
            xp, yp, _ = Cx.shape
            xh = xp//2
            yh = yp//2
            x = np.arange(0, xp-1, 1 - 1e-13)
            y = np.arange(0, yp-1, 1 - 1e-13)
            Y, X = np.meshgrid(y, x)

            #xz

            ax.plot_surface((X-X+yp/2)[:xh, xh:], X[xh:, xh:], Y[:xh, xh:], facecolors=np.rot90(Cz,k=-1)[xh:, xh:],
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)
            
            ax.plot_surface((X-X+yp/2)[:xh+1, xh-1:], X[:xh+1, :xh+1], Y[xh-1:, :xh+1], facecolors=np.rot90(Cz,k=-1)[:xh+1, :xh+1],
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)
                            
            #yz
            
            ax.plot_surface(X[:xh, :xh], (X-X+yp/2)[xh:, xh:], Y[xh:, xh:], facecolors=np.rot90(Cy.transpose((1,0,2)),k=2)[:xh, xh:],
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)
            
            
            ax.plot_surface(X[xh:, :xh], (X-X+yp/2)[xh:, xh:], Y[xh:, :xh], facecolors=np.rot90(Cy.transpose((1,0,2)),k=2)[xh:, :xh],
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)
            
            
            #xy
            ax.plot_surface(X[:xh, :xh], Y[xh:, :xh], (X-X+yp/2)[:xh,:xh], facecolors=np.rot90(Cx,k=1)[:xh,:xh],
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)
            
            ax.plot_surface(X[xh:, xh:], Y[xh:, xh:], (X-X+yp/2)[:xh,:xh], facecolors=np.rot90(Cx,k=1)[xh:,xh:],
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)

            #front
            
            ax.plot_surface((X-X+yp/2)[xh:, xh:], X[:xh, xh:], Y[xh:, xh:], facecolors=np.rot90(Cz,k=-1)[:xh, xh:],
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)
            
            
            ax.plot_surface(X[xh-1:, xh-1:], (X-X+yp/2)[xh-1:, xh-1:], Y[xh-1:, xh-1:], facecolors=np.rot90(Cy.transpose((1,0,2)),k=2)[xh-1:, xh-1:],
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)
            
            
            ax.plot_surface(X[xh:, :xh], Y[:xh, :xh], (X-X+yp/2)[:xh,:xh], facecolors=np.rot90(Cx,k=1)[xh:,:xh],
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)
            
            
        else:
            xp, yp, _ = Cx.shape
            x = np.arange(0, xp, 1 - 1e-13)
            y = np.arange(0, yp, 1 - 1e-13)
            Y, X = np.meshgrid(y, x)

            
            ax.plot_surface(X, Y, X-X+yp, facecolors=np.rot90(Cx,k=1),
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)
            
    
            ax.plot_surface(X, X-X, Y, facecolors=np.rot90(Cy.transpose((1,0,2)),k=2),
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)
            
            ax.plot_surface(X-X+xp, X, Y, facecolors=np.rot90(Cz,k=-1),
                            rstride=1, cstride=1,
                            antialiased=True, shade=False)
            

        if edges:
            ax.plot([xp, xp], [0, xp], xp, **edges_kw)
            ax.plot([0, xp], [0, 0], xp, **edges_kw)
            ax.plot([0, xp], [0, 0], 0, **edges_kw)
            ax.plot([xp, xp], [0, 0], [0, xp], **edges_kw)
            ax.plot([0, 0], [0, 0], [0, xp], **edges_kw)
            ax.plot([xp, xp], [xp, xp], [0, xp], **edges_kw)
            ax.plot([xp, 0],  [xp, xp], xp, **edges_kw)
            ax.plot([0, 0],  [xp, 0], xp, **edges_kw)
            ax.plot([xp, xp],  [0, xp], 0, **edges_kw)

        if coord:
            coord_kw = dict(linewidth=5,linestyle="-", zorder=1e3)
            coord_scale = 1.1
            gap = .05*xp
            ax.plot([xp+gap, coord_scale*xp], [xp, xp], [0, 0],color="r", **coord_kw)
            ax.text(xp*coord_scale+1.5*gap,xp,0,"+X",color='r', fontsize = 18, horizontalalignment='center', verticalalignment='center')

            ax.plot([0, 0], [-gap, xp-coord_scale*xp], [0, 0],color="g", **coord_kw)
            ax.text(0, xp-coord_scale*xp-1.5*gap ,0,"-Z",color='g', fontsize = 18, horizontalalignment='center', verticalalignment='center')

            ax.plot([0, 0], [xp, xp], [xp+gap, coord_scale*xp],color="b", **coord_kw)
            ax.text(0,xp,coord_scale*xp+gap,"+Y",color='b', fontsize = 18, horizontalalignment='center')

        
        if ortho:
            ax.set_proj_type('ortho')

        if cbar:
            m = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
            m.set_array([])
            cbar = fig.colorbar(m, ax=ax,shrink=.6)
            cbar.ax.tick_params(labelsize=18)

        if title is not None:
            ax.set_title(title,loc='left')

        ax.margins(0)
        fig.tight_layout(pad=0)

        if savedir is None:
            buf = io.BytesIO()
            fig.savefig(buf,bbox_inches='tight',pad_inches=0)
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            return img

        else:
            plt.savefig(savedir,transparent=True,dpi=600,bbox_inches='tight',pad_inches=0)
            plt.close()

def plot_grid(im,savedir=None, nrow=5,label=False,labels=None, kwargs_list = None, use_kwargs_list = False, **kwargs):

    images = []
    for i in range(im.shape[0]):
        if label:
            lab = labels[i]
        else:
            lab = None

        if use_kwargs_list:
            kwargs = kwargs_list[i]

        images.append(plot_cube(im[i],title=lab, **kwargs))

    images = np.stack(images,axis=0)
    grid = rearrange(images,"(w r) x y c -> (w x) (r y) c ", w = nrow)

    im = Image.fromarray(np.uint8(grid))
    im.save(savedir)
    



'''
def plot_cube(im,savedir,elev = 30,azim=-45, ortho = True, edges = True, coord=False, edges_kw =  dict(color='k', linewidth=1.5,linestyle="--", zorder=1e3)):

        #rotate the thing to the coord sys we want to visualize in
        im = np.flip(np.rot90(np.rot90(im, k=1, axes=(0, 1)), k=-1, axes=(1, 2)), axis=1)

        Cx = im[0,:,:]
        Cy = im[:,0,:]
        Cz = im[:,:,0]
        

        
        xp, yp, _ = Cx.shape
        x = np.arange(0, xp, 1 - 1e-13)
        y = np.arange(0, yp, 1 - 1e-13)
        Y, X = np.meshgrid(y, x)

        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111, projection='3d')
        ax.dist=6.2
        ax.view_init(elev=elev, azim=azim)
        ax.axis("off")

        ax.plot_surface(X, Y, X-X+yp, facecolors=np.rot90(Cx,k=1),
                        rstride=1, cstride=1,
                        antialiased=True, shade=False)

        ax.plot_surface(X, X-X, Y, facecolors=np.rot90(Cy.transpose((1,0,2)),k=2),
                        rstride=1, cstride=1,
                        antialiased=True, shade=False)

        ax.plot_surface(X-X+xp, X, Y, facecolors=np.rot90(Cz,k=-1),
                        rstride=1, cstride=1,
                        antialiased=True, shade=False)
        
        
        if edges:
            ax.plot([xp, xp], [0, xp], xp, **edges_kw)
            ax.plot([0, xp], [0, 0], xp, **edges_kw)
            ax.plot([0, xp], [0, 0], 0, **edges_kw)
            ax.plot([xp, xp], [0, 0], [0, xp], **edges_kw)
            ax.plot([0, 0], [0, 0], [0, xp], **edges_kw)
            ax.plot([xp, xp], [xp, xp], [0, xp], **edges_kw)
            ax.plot([xp, 0],  [xp, xp], xp, **edges_kw)
            ax.plot([0, 0],  [xp, 0], xp, **edges_kw)
            ax.plot([xp, xp],  [0, xp], 0, **edges_kw)

        if coord:
            coord_kw = dict(linewidth=5,linestyle="-", zorder=1e3)
            coord_scale = 1.1
            gap = .05*xp
            ax.plot([xp+gap, coord_scale*xp], [xp, xp], [0, 0],color="r", **coord_kw)
            ax.text(xp*coord_scale+1.5*gap,xp,0,"X",color='r', fontsize = 18, horizontalalignment='center', verticalalignment='center')

            ax.plot([0, 0], [-gap, xp-coord_scale*xp], [0, 0],color="g", **coord_kw)
            ax.text(0, xp-coord_scale*xp-1.5*gap ,0,"Z",color='g', fontsize = 18, horizontalalignment='center', verticalalignment='center')

            ax.plot([0, 0], [xp, xp], [xp+gap, coord_scale*xp],color="b", **coord_kw)
            ax.text(0,xp,coord_scale*xp+gap,"Y",color='b', fontsize = 18, horizontalalignment='center')

        
        if ortho:
            ax.set_proj_type('ortho')
        fig.tight_layout()
        
        plt.savefig(savedir,transparent=True,dpi=300)
        plt.close()


from orix import plot
from orix.quaternion import symmetry, Orientation, Quaternion
pg432 = symmetry.O
ipf_key = plot.IPFColorKeyTSL(pg432)

def pc_ipf(euler, path):
    shape = euler.shape
    ori = Orientation.from_euler(euler.reshape(-1,3), pg432, degrees=False)
    ori = ori.map_into_symmetry_reduced_zone()
    rgb = ipf_key.orientation2color(ori).reshape(shape)
    plot_cube(rgb, path)


def Cubeplot( im, #image array (x,y,z) for scalar or (x,y,z,3) for RGB, (x,y,z,4) for RGBA
              savedir,
              mode  = "scalar", #scalar or rgb,
              type  = "surface", #surface: surface of volume, interior: for 3 orthogonal slices through the center
              title = None,
              elev  = 30,
              azim  =-45, 
              ortho = True,  #Orthographic or perspective plotting
              edges = True,  #Plot bounding box edges
              coord = True, #Plot coordinate system ref
              cmap  = "viridis", #Cmap for scalar mode
              cbar  = False,  #add a colorbar,
              dpi = 300,
              edges_kw =  dict(color='k', linewidth=1.5,linestyle="--", zorder=1e3) #line style setting 
              ):

    #rotate the thing to the coord sys we want to visualize in
    im = np.flip(np.rot90(np.rot90(im, k=1, axes=(0, 1)), k=-1, axes=(1, 2)), axis=1)

    if mode == "rgb":
        if im.shape[3] == 3:
            alpha = np.ones((im.shape[0],im.shape[1],im.shape[2],1))
            im = np.concatenate((im,alpha),axis=3)

    if mode == "scalar":
        vmin = im.min()
        vmax = im.max()

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        im = plt.get_cmap(cmap)(norm(im))


    if type == "surface":
        Cx = im[0,:,:]
        Cy = im[:,0,:]
        Cz = im[:,:,0]

        Cx_b = im[-1, :, :]
        Cy_b = im[:, -1, :]
        Cz_b = im[:, :, -1]

    if type == "interior":
        center = np.array(im.shape)//2
        Cx = im[center[0],:,:]
        Cy = im[:,center[1],:]
        Cz = im[:,:,center[2]]
    
    
    xp, yp, _ = Cx.shape
    x = np.arange(0, xp, 1 - 1e-13)
    y = np.arange(0, yp, 1 - 1e-13)
    Y, X = np.meshgrid(y, x)

    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.dist=6.2
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")

    # plot one plane
    def plot_plane(xx, yy, zz, col):
        ax.plot_surface(
            xx,
            yy,
            zz,
            facecolors=col,
            rstride=1,
            cstride=1,
            antialiased=True,
            shade=False,
        )

    if type == "surface":
        # Z=const
        plot_plane(X, Y, 0 * X + yp, np.rot90(Cx_b, k=1))
        plot_plane(X, Y, 0 * X, np.rot90(Cx, k=1))
        # Y=const
        plot_plane(X, 0 * X, Y, np.rot90(Cy_b.transpose((1, 0, 2)), k=2))
        plot_plane(X, 0 * X + yp, Y, np.rot90(Cy.transpose((1, 0, 2)), k=2))
        # X = const
        plot_plane(0 * X + xp, X, Y, np.rot90(Cz_b, k=-1))
        plot_plane(0 * X, X, Y, np.rot90(Cz, k=-1))
    
    if type == "interior":
        assert 0 == 1, "NOT CODED YET"
    
    if edges:
        ax.plot([xp, xp], [0, xp], xp, **edges_kw)
        ax.plot([0, xp], [0, 0], xp, **edges_kw)
        ax.plot([0, xp], [0, 0], 0, **edges_kw)
        ax.plot([xp, xp], [0, 0], [0, xp], **edges_kw)
        ax.plot([0, 0], [0, 0], [0, xp], **edges_kw)
        ax.plot([xp, xp], [xp, xp], [0, xp], **edges_kw)
        ax.plot([xp, 0],  [xp, xp], xp, **edges_kw)
        ax.plot([0, 0],  [xp, 0], xp, **edges_kw)
        ax.plot([xp, xp],  [0, xp], 0, **edges_kw)

    if coord:
        coord_kw = dict(linewidth=5,linestyle="-", zorder=1e3)
        coord_scale = 1.1
        gap = .05*xp
        ax.plot([xp+gap, coord_scale*xp], [xp, xp], [0, 0],color="r", **coord_kw)
        ax.text(xp*coord_scale+1.5*gap,xp,0,"X",color='r', fontsize = 18, horizontalalignment='center', verticalalignment='center')

        ax.plot([0, 0], [-gap, xp-coord_scale*xp], [0, 0],color="g", **coord_kw)
        ax.text(0, xp-coord_scale*xp-1.5*gap ,0,"Z",color='g', fontsize = 18, horizontalalignment='center', verticalalignment='center')

        ax.plot([0, 0], [xp, xp], [xp+gap, coord_scale*xp],color="b", **coord_kw)
        ax.text(0,xp,coord_scale*xp+gap,"Y",color='b', fontsize = 18, horizontalalignment='center')

    
    if ortho:
        ax.set_proj_type('ortho')
    fig.tight_layout()

    if cbar:
        m = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        m.set_array([])
        fig.colorbar(m, ax=ax)
    
    if title is not None:
        ax.set_title(title)
    
    plt.savefig(savedir,transparent=True,dpi=dpi)
    plt.close()
'''



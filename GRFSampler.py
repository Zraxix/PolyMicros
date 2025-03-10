import torch
ctol = 1e-8

def StatisticsGenerator(stats, f1 = None, zmean=True, valid_check = True):
    '''
    StatisticsGeneratorPoly
    Written by Andreas Robertson and Michael Buzzy

    Ref: https://doi.org/10.1016/j.actamat.2022.117927

    Description: Efficent Sampling of GRF defined by the 2ps 

    args:
        stats: Array of Stastics: Shape: (Spatial 1, Sptaial 2, ...,  n_stats) 
        f1: Constant to specify mean. None is structure is strictly positive. Used Float

        valid_check: If true runs several tests to ensure parameters 
                     are formatted correctly and in the proper range. 

    returns:
        sample  : Sampled Structure  (Spatial 1, Sptaial 2, ...,  n_phases) 

    '''
    dev = stats.device
    shape = torch.tensor(stats.shape[:-1])
    N = torch.prod(shape)
    dim = len(shape)
    shape = shape.tolist()

    stats_fft = torch.fft.fftn(stats, dim = tuple(range(0,dim)))

    if f1:
        indx1 = tuple([0] * (dim) + [slice(None)])
        indx2 = tuple([0] * (dim+1))
        means = (stats_fft[indx1].conj() * f1 / (stats_fft[indx2])).real
    else:
        means = stats_fft[tuple([0] * (dim) + [slice(None)])].real
        means[0] = (means[0]/N)**(0.5) #true if mean is positive
        means[1:] = (means[1:] / N) / means[0]

        #print(means[0])



    interfilter = torch.ones_like(stats_fft)
    interfilter[..., 1:] = stats_fft[..., 1:] / (stats_fft[..., 0, None] + ctol)

    eigs = stats_fft[...,0].real
    eigs[tuple([0] * dim)] = 0.0

    eigs = eigs / N

    if eigs.min() < -ctol and valid_check:
        raise ValueError('The autocovariance contains at least one negative eigenvalue (' + str(eigs.min()) + ').')
    
    eigs[eigs < 0.0] = 0.0

    eigs = torch.sqrt(eigs)
    eps = torch.randn(shape) + 1j * torch.randn(shape)
    eps = eps.to(dev)
    new = torch.fft.fftn(eigs * eps)

    if zmean:
        s1 = new.real
    else:
        s1 = new.real + means[0]
        #s2 = new.imag + means[0]

    struct1 = torch.fft.ifftn(
        torch.fft.fftn(s1)[..., None]*interfilter,
        dim=tuple(range(0,dim))
    ).real

    return struct1

def crosscorrelation(arr1, arr2):
    '''
    StatisticsGeneratorPoly
    Written by Andreas Robertson and Michael Buzzy

    Description: Compute Crosscorrelation (or autocorrelation if arr1 == arr2)

    args:
        arr1: Array of Local State: Shape: (Spatial 1, Sptaial 2, ...) 
        arr2: Array of Local State: Shape: (Spatial 1, Sptaial 2, ...) 

    returns:
        sample  : Statistics (Spatial 1, Sptaial 2, ...)
    '''
    ax = list(range(0, len(arr1.shape)))
    arr1_FFT = torch.fft.fftn(arr1, dim=ax)
    arr2_FFT = torch.fft.fftn(arr2, dim=ax)
    return torch.fft.ifftn(arr1_FFT.conj() * arr2_FFT, s=arr1.shape, dim=ax).real / torch.prod(
        torch.tensor(arr1.shape))

def TwoPCorrelation(struct):
    '''
    Struct Shape: (... Spatial Dims ..., Channel)
    '''
    return torch.stack([crosscorrelation(struct[...,0],struct[...,j])  for j in range(3)],-1)

if __name__ == "__main__":
    print("File run as main. Not Imported. Running Tests!")
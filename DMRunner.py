import torch
import h5py

from diffusion import Diffusion

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from HDFPartialLoader import HDF5PartialDataset

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.profilers import AdvancedProfiler

import datetime
import json
import argparse
import os
import random 
import glob

torch.set_float32_matmul_precision('high')

def dict2namespace(config: dict):
    """
    Converts a dictionary to a namespace
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace



#set up termianl interface
parser = argparse.ArgumentParser(
                    prog='DMRunner',
                    description='Handles Training and Inference for Diffusion Models ',
                    epilog='Written by Michael Buzzy')

parser.add_argument('mode',type=str, help="Mode to run script in. One of: train, sample, plot")
parser.add_argument('-c','--config')
parser.add_argument('-d','--dataset')
parser.add_argument('-m','--model')
parser.add_argument('-s','--steps', type=int)
parser.add_argument('-e','--edge_length', type=int)
parser.add_argument('-p','--prefix', nargs='*',default="",help="HDF5 Group or dataset to load for training, can be multiple if a dataset.")
parser.add_argument('-i','--inital',help="initalization for diffusion sampling. if none initalization is random.")
parser.add_argument('--skip', type=int, help="When sampling the numner of diffusion steps to skip at the start of the process. Typically used with an initalization")
parser.add_argument('--nsamples',type=int,default=1)
parser.add_argument('--dir', default=None)
parser.add_argument('--HDFPartial', action='store_true',help="If each sample is stored in a seperate hdf5 dataset under the prefix group this flag will only load a single batch of training data at a time to conserve cpu memory. Otherwise will load the prefix as if it is a single dataset with all the samples.")

args = parser.parse_args()


if args.dir is not None:
    args.model = glob.glob(args.dir+"/logs/version_0/checkpoints/*")[-1]
    #print(args.model)
    args.config = args.dir+"/config.json"


#load config
with open(args.config,'r') as f:
    config = json.load(f)
    config = dict2namespace(config)

if config.dims == 3:
    if config.periodic:
        from UnetP import UnetND
    else:
        from Unet import UnetND

elif config.dims == 2:
    print("2D not implimented yet in this version")
    exit()

else:
    exit()

if config.precision == "16-mixed":
        dtype = torch.half
else:
        dtype = torch.float

if args.mode == "train":
    #set up training dir
    config.id = "DM"+"_"+args.prefix[0]+"_"+datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + "_R" + str(random.randint(0,9))
    exp_dir = "experiments/"+config.id+"/"

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    #add data to config and save
    config.dataset = args.dataset

    with open(exp_dir+"config.json",'w') as f:
        json.dump(vars(config),f)
    
    #load dataset
    dataloaders = []

    if not isinstance(args.prefix,list):
            args.prefix = [args.prefix]

    if args.HDFPartial:
        for prefix in args.prefix:
            dataset = HDF5PartialDataset(path = args.dataset, prefix = prefix)
            dataloaders.append(
                            DataLoader(dataset, batch_size=config.batch_size, shuffle=True,num_workers=12)
                        )
    else:
        with h5py.File(args.dataset, "r") as dat:
            for prefix in args.prefix:
                dataset = TensorDataset(torch.tensor(dat[prefix][:],dtype=dtype))
                dataloaders.append(
                        DataLoader(dataset, batch_size=config.batch_size, shuffle=True,num_workers=6)
                    )

    unet = UnetND(
        config.udim,
        dim_mults= tuple(config.channels),
        channels=config.in_channels
    )
        
    unet.sample_size = config.sample_size
    unet.in_channels = config.in_channels
    unet.dims=config.dims

    if config.compile:
        unet = torch.compile(unet)

    logger = CSVLogger(exp_dir, name="logs")

    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(dirpath=exp_dir+"/logs/version_0/checkpoints", every_n_epochs=2,save_top_k=-1, save_weights_only=True)

    trainer = Trainer(max_epochs=config.epochs,logger=logger, 
                      accelerator="auto", 
                      accumulate_grad_batches=config.accumulate_grad_batches, 
                      precision=config.precision,
                      callbacks=[lr_monitor,checkpoint_callback],
                      profiler="simple",
                      gradient_clip_val = config.gradient_clip_val)

    if args.model is not None:
       diffusion = Diffusion.load_from_checkpoint(
        args.model,
        model=unet,
        samples_path =exp_dir+"samples/",
        bins_max = config.sample_steps,
        epochs_max = config.epochs
       )
    else: 
        diffusion = Diffusion(
        model=unet,
        learning_rate = config.learning_rate,
        samples_path =exp_dir+"samples/",
        bins_max = config.sample_steps,
        epochs_max = config.epochs
    )
        
    
    trainer.fit(diffusion, dataloaders)

if args.mode == "sample":

    unet = UnetND(
        config.udim,
        dim_mults= tuple(config.channels),
        channels=config.in_channels
    )
        
    unet.sample_size = args.edge_length
    unet.in_channels = config.in_channels
    unet.dims=config.dims

    if config.compile:
        unet = torch.compile(unet)

    diffusion = Diffusion.load_from_checkpoint(args.model,model=unet, bins_max = config.sample_steps)

    if args.inital == None:
        diffusion.save_samples(filename="samples",num_samples = args.nsamples, steps = args.steps,  save_numpy = True)
    else:
        with h5py.File(args.inital, "r") as dat:
            inital = torch.tensor(dat[args.prefix[0]],dtype=dtype)

           
        diffusion.save_samples(filename="samples",num_samples = args.nsamples, steps = args.steps, images = inital, skip = args.skip, save_numpy = True)

if args.mode == "plot":

    with h5py.File(args.inital, "r") as dat:
        inital = np.array(dat[args.prefix[0]])

    if len(inital.shape) == 5:
        Diffusion.plot_cube(inital[0]/2 + .5, "samples/plot.png")
    else:
        Diffusion.plot_slice(inital[0]/2 + .5, "samples/plot.png")







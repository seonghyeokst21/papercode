
import argparse
import torch
import numpy as np, h5py

import os
import torch.optim as optim
import torchvision
from backbones.ncsnpp_generator_adagn import NCSNpp
from dataset_brain_test import CreateDatasetSynthesis
from skimage.metrics import peak_signal_noise_ratio as psnr  # Import from scikit-image
from skimage.metrics import structural_similarity as ssim
from utils.utils import *

import torch.nn.functional as F

import torchvision.transforms as transforms

def compute_ssim(img1, img2):
    # Convert torch tensors to numpy arrays and move them to CPU
    img1_np = img1.squeeze().cpu().numpy()
    img2_np = img2.squeeze().cpu().numpy()
    
    # Calculate the range manually based on the image range
    data_range = img2_np.max() - img2_np.min()
    
    # Calculate SSIM with data_range specified
    return ssim(img1_np, img2_np, data_range=data_range)

def load_checkpoint(checkpoint_dir, netG, name_of_network, epoch,device = 'cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)  

    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint
   
    for key in list(ckpt.keys()):
         ckpt[key[7:]] = ckpt.pop(key)
    # netG.load_state_dict(ckpt)
    netG.load_state_dict(ckpt, strict=False)

    netG.eval()
#%%
def sample_and_test(args):
    torch.manual_seed(42)
    # device = 'cuda:0'
    torch.cuda.set_device(args.gpu_chose)
    device = torch.device('cuda:{}'.format(args.gpu_chose))
    epoch_chosen=args.which_epoch
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    #loading dataset
    phase='test'
    dataset, paths = CreateDatasetSynthesis(phase, args.input_path, args.contrast1, args.contrast2)
    data_loader = torch.utils.data.DataLoader(list(zip(dataset, paths)), batch_size=1, shuffle=False, num_workers=4)

    #Initializing and loading network
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)

    exp = args.exp
    output_dir = args.output_path
    exp_path = os.path.join(output_dir,exp)

    checkpoint_file = exp_path + "/{}_{}.pth"
    load_checkpoint(checkpoint_file, gen_diffusive_1,'gen_diffusive_1',epoch=str(epoch_chosen), device = device)
    load_checkpoint(checkpoint_file, gen_diffusive_2,'gen_diffusive_2',epoch=str(epoch_chosen), device = device)

    T = get_time_schedule(args, device)
    
    pos_coeff = Posterior_Coefficients(args, device)
         
    save_dir = exp_path + "/generated_samples/epoch_{}_ct2t2_visual".format(epoch_chosen)
    
    loss1 = np.zeros((1,len(data_loader)))
    loss2 = np.zeros((1,len(data_loader)))
    ssim_loss1 = np.zeros((1, len(data_loader)))
    ssim_loss2 = np.zeros((1, len(data_loader)))

    for iteration, (data, paths) in enumerate(data_loader):
        x, y = data
        x_path, y_path = paths

        patient_number_x = os.path.basename(os.path.dirname(x_path[0]))
        patient_number_y = os.path.basename(os.path.dirname(y_path[0]))

        real_data = y.to(device, non_blocking=True)
        source_data = x.to(device, non_blocking=True)
        
        x2_t = torch.cat((torch.randn_like(real_data),source_data),axis=1)

        fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)
        fake_sample2 = to_range_0_1(fake_sample2) ; fake_sample2 = fake_sample2/fake_sample2.max()
        real_data = to_range_0_1(real_data) ; real_data = real_data/real_data.max()
        source_data = to_range_0_1(source_data); source_data = source_data/source_data.max() 
        
        filename_x = os.path.basename(x_path[0])
        filename_y = os.path.basename(y_path[0])
        
        patient_save_dir_x = os.path.join(save_dir, patient_number_x)
        patient_save_dir_y = os.path.join(save_dir, patient_number_y)
        
        if not os.path.exists(patient_save_dir_x+filename_y[:2]):
            os.makedirs(patient_save_dir_x+filename_y[:2])
        if not os.path.exists(patient_save_dir_y+filename_y[:2]):
            os.makedirs(patient_save_dir_y+filename_y[:2])

        loss2[0, iteration] = psnr(fake_sample2.squeeze().cpu().numpy(), real_data.squeeze().cpu().numpy())
        ssim_loss2[0, iteration] = compute_ssim(fake_sample2, real_data)

        print(str(iteration))
        print(loss2[0, iteration], ssim_loss2[0, iteration])

        # torchvision.utils.save_image(fake_sample2, os.path.join(patient_save_dir_y+filename_y[:8], f"{filename_y[:-4]}.png"[1:]), normalize=True)
        fake_sample2 = torch.cat((source_data, fake_sample2, real_data),axis=-1)
        torchvision.utils.save_image(fake_sample2, os.path.join(patient_save_dir_y+filename_y[:2], f"{filename_y[:-4]}.png"[3:]), normalize=True)

    mean_loss2 = round(np.nanmean(loss2), 4)
    std_loss2 = round(np.nanstd(loss2), 4)
    mean_ssim_loss2 = round(np.nanmean(ssim_loss2), 4)
    std_ssim_loss2 = round(np.nanstd(ssim_loss2), 4)
    print(f"PSNR Mean±std: {mean_loss2} ± {std_loss2}")
    print(f"SSIM Mean±std: {mean_ssim_loss2} ± {std_ssim_loss2}")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--num_channels', type=int, default=2,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=32,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=1,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='exp_syndiff', help='name of experiment')
    parser.add_argument('--input_path', help='path to input data')
    parser.add_argument('--output_path', help='path to output saves')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='sample generating batch size')
    
    #optimizaer parameters    
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--contrast1', type=str, default='T1',
                        help='contrast selection for model')
    parser.add_argument('--contrast2', type=str, default='T2',
                        help='contrast selection for model')
    parser.add_argument('--which_epoch', type=int, default=50)
    parser.add_argument('--gpu_chose', type=int, default=0)


    parser.add_argument('--source', type=str, default='T2',
                        help='source contrast')   
    args = parser.parse_args()
    
    sample_and_test(args)

# python3 test.py --image_size 512 --ch_mult 1 1 2 2 4 4 --contrast1 CT --contrast2 T1 --input_path ../datasets/kumc_invisible/test --output_path ./output_kumc_duct/for/results_1015 --num_channels_dae 8 --which_epoch 120

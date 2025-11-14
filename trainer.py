import argparse
import torch
import numpy as np
import os
import shutil
import time
from torch.multiprocessing import Process
import torch.distributed as dist
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.nn import functional as F
import torch.nn as nn
from torch import optim
import torchvision
from dataset import CreateDatasetSynthesis
import pywt
from utils.utils import *
from utils.utils2 import *
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from torch.utils.tensorboard import SummaryWriter
import torch.profiler

### ### ### ### ### Train Adversarial Diffusion Model ### ### ### ### ###
def train_syndiff(rank, gpu, args):
    
    from backbones.discriminator_noiseless import Discriminator_small, Discriminator_large
    
    from backbones.ncsnpp_generator_adagn import NCSNpp
    
    import backbones.generator_resnet 

    from utils.EMA import EMA
    
    #rank = args.node_rank * args.num_process_per_node + gpu

    plot_save_path = './plots'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = args.batch_size
    
    nz = args.nz #latent dimension
    
    ### Create the train_dataset (train & validation) ###
    # train_dataset = CreateDatasetSynthesis(phase = "train", input_path = args.input_path, contrast1 = args.contrast1, contrast2 = args.contrast2)
    # val_dataset = CreateDatasetSynthesis(phase = "val", input_path = args.input_path, contrast1 = args.contrast1, contrast2 = args.contrast2)

    cache_dir = "./dataset_cache"
    os.makedirs(cache_dir, exist_ok=True)

    exp = args.exp
    output_path = args.output_path

    exp_path = os.path.join(output_path,exp)
    save_dir = os.path.join(exp_path, "val_images")
    debug_dir = os.path.join(exp_path, "debug_images")


    # ✅ TensorBoard Writer 초기화 추가
    if rank == 0:
        log_dir = os.path.join(exp_path, 'tensorboard_logs')
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('./backbones', os.path.join(exp_path, 'backbones'))

    train_cache_file = os.path.join(cache_dir, f"train_{args.contrast1}_{args.contrast2}.pkl")
    val_cache_file = os.path.join(cache_dir, f"val_{args.contrast1}_{args.contrast2}.pkl")

    # Train dataset 로드/생성
    if rank == 0:
        if os.path.exists(train_cache_file):
            print('pass')
        else:
            print("Creating train dataset...")
            train_dataset = CreateDatasetSynthesis(phase="train", input_path=args.input_path, 
                                                contrast1=args.contrast1, contrast2=args.contrast2)
            print("Saving train dataset to cache...")
            with open(train_cache_file, 'wb') as f:
                pickle.dump(train_dataset, f)
            print("Train dataset cached!")

        # Val dataset 로드/생성
        if os.path.exists(val_cache_file):
            print('pass')
        else:
            print("Creating val dataset...")
            val_dataset = CreateDatasetSynthesis(phase="val", input_path=args.input_path, 
                                            contrast1=args.contrast1, contrast2=args.contrast2)
            print("Saving val dataset to cache...")
            with open(val_cache_file, 'wb') as f:
                pickle.dump(val_dataset, f)
            print("Val dataset cached!")
    dist.barrier()

    with open(train_cache_file, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(val_cache_file, 'rb') as f:
        val_dataset = pickle.load(f)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=val_sampler,
                                               drop_last = True)

    ### Initializes L1 loss & PSNR in validation ###
    val_l1_loss=np.zeros([2,args.num_epoch,len(val_data_loader)])
    val_psnr_values=np.zeros([2,args.num_epoch,len(val_data_loader)])
    print('train data size:'+str(len(train_data_loader)))
    print('val data size:'+str(len(val_data_loader)))
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # +-----------------------------+
    # |   Initializes the Models !  |
    # +-----------------------------+
    ### Generators performing reverse denoising ###
    
    gen_diffusive_1 = SimplifiedUNet(args).to(device)
    gen_diffusive_2 = SimplifiedUNet(args).to(device)

    ### Generators performing translation for unpaired dataset ###
    args.num_channels=1

    ### Discriminator discriminating real_x_t-k or fake_x_t-k(denoised) ###
    disc_diffusive_1 = Discriminator_small(nc = 1, ngf = args.ngf,
                                   act=nn.LeakyReLU(0.2)).to(device)
    disc_diffusive_2 = Discriminator_small(nc = 1, ngf = args.ngf, 
                                    act=nn.LeakyReLU(0.2)).to(device)
    
    conditional_disc_diffusive_1 = Discriminator_small(nc = 1, ngf = args.ngf,
                                   act=nn.LeakyReLU(0.2)).to(device)
    conditional_disc_diffusive_2 = Discriminator_small(nc = 1, ngf = args.ngf, 
                                    act=nn.LeakyReLU(0.2)).to(device)
    
    # # ✅ 모델 구조 로깅 (rank 0만)
    # if rank == 0:
    #     # Dummy input으로 모델 그래프 그리기
    #     dummy_img = torch.randn(1, 1, 256, 256).to(device)  # [B, C, H, W]
    #     dummy_latent = torch.randn(1, 8, 64, 64).to(device)  # [B, 8, 64, 64] - Generator input
    #     dummy_t = torch.randint(0, args.num_timesteps, (1,), device=device)
    #     dummy_z = torch.randn(1, args.nz, device=device)
        
    #     try:
    #         # Discriminator graph
    #         writer.add_graph(disc_diffusive_1, dummy_img)
    #         print("✅ Discriminator graph logged to TensorBoard")
            
    #         # Generator graph (DDP 전에)
    #         writer.add_graph(gen_diffusive_1, (dummy_latent, dummy_t, dummy_z))
    #         print("✅ Generator graph logged to TensorBoard")
    #     except Exception as e:
    #         print(f"⚠️ Could not log model graph: {e}")
        
    #     # 모델 파라미터 수 출력
    #     total_params_gen = sum(p.numel() for p in gen_diffusive_1.parameters())
    #     total_params_disc = sum(p.numel() for p in disc_diffusive_1.parameters())
    #     print(f"Generator parameters: {total_params_gen:,}")
    #     print(f"Discriminator parameters: {total_params_disc:,}")
        
    #     # TensorBoard에 기록
    #     writer.add_text('Model/Generator_params', f'{total_params_gen:,}')
    #     writer.add_text('Model/Discriminator_params', f'{total_params_disc:,}')

    ### Broadcasting the models ###
    broadcast_params(gen_diffusive_1.parameters())
    broadcast_params(gen_diffusive_2.parameters())
    broadcast_params(disc_diffusive_1.parameters())
    broadcast_params(disc_diffusive_2.parameters())
    broadcast_params(conditional_disc_diffusive_1.parameters())
    broadcast_params(conditional_disc_diffusive_2.parameters())
    
    ### ### ### --- Initializes the Models' Optimizer --- ### ### ###
    ### Generators' Optimizer ###
    optimizer_gen_diffusive_1 = optim.Adam(gen_diffusive_1.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    optimizer_gen_diffusive_2 = optim.Adam(gen_diffusive_2.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    ### Discriminators' Optimizer ###
    optimizer_disc_diffusive_1 = optim.Adam(disc_diffusive_1.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizer_disc_diffusive_2 = optim.Adam(disc_diffusive_2.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))

    optimizer_conditional_disc_diffusive_1 = optim.Adam(conditional_disc_diffusive_1.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizer_conditional_disc_diffusive_2 = optim.Adam(conditional_disc_diffusive_2.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))

    if args.use_ema:
        optimizer_gen_diffusive_1 = EMA(optimizer_gen_diffusive_1, ema_decay=args.ema_decay)
        optimizer_gen_diffusive_2 = EMA(optimizer_gen_diffusive_2, ema_decay=args.ema_decay)

    ### ### ### --- Initializes the Models' Scheduler --- ### ### ###
    ### Generators' Scheduler ###
    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_gen_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_2, args.num_epoch, eta_min=1e-5)

    ### Discriminators' Scheduler ###
    scheduler_disc_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_2, args.num_epoch, eta_min=1e-5)

    scheduler_conditional_disc_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_conditional_disc_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_conditional_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_conditional_disc_diffusive_2, args.num_epoch, eta_min=1e-5)

    ### ### ### --- Initializes DDP --- ### ### ###
    gen_diffusive_1 = nn.parallel.DistributedDataParallel(gen_diffusive_1, device_ids=[gpu])
    gen_diffusive_2 = nn.parallel.DistributedDataParallel(gen_diffusive_2, device_ids=[gpu])
    disc_diffusive_1 = nn.parallel.DistributedDataParallel(disc_diffusive_1, device_ids=[gpu])
    disc_diffusive_2 = nn.parallel.DistributedDataParallel(disc_diffusive_2, device_ids=[gpu])
    conditional_disc_diffusive_1 = nn.parallel.DistributedDataParallel(conditional_disc_diffusive_1, device_ids=[gpu])
    conditional_disc_diffusive_2 = nn.parallel.DistributedDataParallel(conditional_disc_diffusive_2, device_ids=[gpu])


    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('./backbones', os.path.join(exp_path, 'backbones'))

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    

    ### ### ### --- Optional Re-train --- ### ### ###
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        gen_diffusive_1.load_state_dict(checkpoint['gen_diffusive_1_dict'])
        gen_diffusive_2.load_state_dict(checkpoint['gen_diffusive_2_dict'])
        
        # load G        
        optimizer_gen_diffusive_1.load_state_dict(checkpoint['optimizer_gen_diffusive_1'])
        scheduler_gen_diffusive_1.load_state_dict(checkpoint['scheduler_gen_diffusive_1'])
        optimizer_gen_diffusive_2.load_state_dict(checkpoint['optimizer_gen_diffusive_2'])
        scheduler_gen_diffusive_2.load_state_dict(checkpoint['scheduler_gen_diffusive_2']) 

        # load D
        disc_diffusive_1.load_state_dict(checkpoint['disc_diffusive_1_dict'])
        optimizer_disc_diffusive_1.load_state_dict(checkpoint['optimizer_disc_diffusive_1'])
        scheduler_disc_diffusive_1.load_state_dict(checkpoint['scheduler_disc_diffusive_1'])

        disc_diffusive_2.load_state_dict(checkpoint['disc_diffusive_2_dict'])
        optimizer_disc_diffusive_2.load_state_dict(checkpoint['optimizer_disc_diffusive_2'])
        scheduler_disc_diffusive_2.load_state_dict(checkpoint['scheduler_disc_diffusive_2'])   

        conditional_disc_diffusive_1.load_state_dict(checkpoint['conditional_disc_diffusive_1_dict'])
        optimizer_conditional_disc_diffusive_1.load_state_dict(checkpoint['optimizer_conditional_disc_diffusive_1'])
        scheduler_conditional_disc_diffusive_1.load_state_dict(checkpoint['scheduler_conditional_disc_diffusive_1'])

        conditional_disc_diffusive_2.load_state_dict(checkpoint['conditional_disc_diffusive_2_dict'])
        optimizer_conditional_disc_diffusive_2.load_state_dict(checkpoint['optimizer_conditional_disc_diffusive_2'])
        scheduler_conditional_disc_diffusive_2.load_state_dict(checkpoint['scheduler_conditional_disc_diffusive_2'])   

        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    

    ### add code ### - seonghyeok's 
    vae = load_trained_vae('vae_ct_mri_shared.pt')
    print_gpu_memory("Before loading VAE")
    # +------------------+
    # |  Start Training  |
    # +------------------+
    for epoch in range(init_epoch, args.num_epoch):
        print(epoch)
        wavelet_loss_syn_values = []
        wavelet_loss_cyc_values = []
        lambda_cGAN_values = []
        lambda_cycleGAN_values = []

        ### Sampling different subsets of the dataset in each epoch ###
        train_sampler.set_epoch(epoch)
        for iteration, (x1, x2) in enumerate(train_data_loader):
            for p in disc_diffusive_1.parameters():
                p.requires_grad = True
            for p in disc_diffusive_2.parameters():
                p.requires_grad = True
            for p in conditional_disc_diffusive_1.parameters():
                p.requires_grad = True
            for p in conditional_disc_diffusive_2.parameters():
                p.requires_grad = True

            ### update the gradients to 0 ###
            disc_diffusive_1.zero_grad()
            disc_diffusive_2.zero_grad()
            conditional_disc_diffusive_1.zero_grad()
            conditional_disc_diffusive_2.zero_grad()

            optimizer_disc_diffusive_1.zero_grad()
            optimizer_disc_diffusive_2.zero_grad()
            optimizer_conditional_disc_diffusive_1.zero_grad()
            optimizer_conditional_disc_diffusive_2.zero_grad()

            ### sample from Original Sample ###
            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)
            
            ### Sample t ###
            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            
            fixed_t1 = torch.full((real_data1.size(0),), args.num_timesteps - 1, device=device, dtype=torch.int64)
            fixed_t2 = torch.full((real_data2.size(0),), args.num_timesteps - 1, device=device, dtype=torch.int64)

            ### encoding ### -seonghyeok's 
            l1 = encode_images(vae, real_data1)
            l2 = encode_images(vae, real_data2)

            ### sample x_t and x_{t+1} ###
            x1_t, x1_tp1 = q_sample_pairs(coeff, l1, t1)
            #x1_t.requires_grad = True
            
            x2_t, x2_tp1 = q_sample_pairs(coeff, l2, t2)
            #x2_t.requires_grad = True

            fixed_x1_t, fixed_x1_tp1 = q_sample_pairs(coeff, l1, fixed_t1)
            #fixed_x1_t.requires_grad = True
            
            fixed_x2_t, fixed_x2_tp1 = q_sample_pairs(coeff, l2, fixed_t2)
            #fixed_x2_t.requires_grad = True
            print_gpu_memory("Train 1")
            # # +--------------------------+
            # # |  Discriminator Training  |
            # # +--------------------------+
            # ### training discriminator with real sample ###
            # D1_real = disc_diffusive_1(real_data1).view(-1)
            # D2_real = disc_diffusive_2(real_data2).view(-1)
            
            # cD1_real = conditional_disc_diffusive_1(real_data1).view(-1)
            # cD2_real = conditional_disc_diffusive_2(real_data2).view(-1)

            # ### calculates real error in Discriminator ###
            
            # # Evaluates how well the discriminator distinguishes "real data" : F.softplus(-D{1,2}_real)
            # errD1_real = F.softplus(-D1_real)
            # errD1_real = errD1_real.mean()
            
            # errD2_real = F.softplus(-D2_real)
            # errD2_real = errD2_real.mean()
            # errD_real = errD1_real + errD2_real

            # ### conditional ###
            # err_cD1_real = F.softplus(-cD1_real)
            # err_cD1_real = err_cD1_real.mean()
            
            # err_cD2_real = F.softplus(-cD2_real)
            # err_cD2_real = err_cD2_real.mean()
            # err_cD_real = err_cD1_real + err_cD2_real

            # ### apply the gradient penalty in each epoch ###
            # if args.lazy_reg is None:
            #     grad1_real = torch.autograd.grad(
            #                 outputs=D1_real.sum(), inputs=decode_latents(vae,x1_t), create_graph=True
            #                 )[0]
            #     grad1_penalty = (
            #                     grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2
            #                     ).mean()
            #     grad2_real = torch.autograd.grad(
            #                 outputs=D2_real.sum(), inputs=decode_latents(vae,x2_t), create_graph=True
            #                 )[0]
            #     grad2_penalty = (
            #                     grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2
            #                     ).mean()
                
            #     grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
            #     grad_penalty.backward()
            # else:
            #     if global_step % args.lazy_reg == 0:
            #         grad1_real = torch.autograd.grad(
            #                 outputs=D1_real.sum(), inputs=decode_latents(vae,x1_t), create_graph=True, allow_unused=True
            #                 )[0]
            #         grad2_real = torch.autograd.grad(
            #                 outputs=D2_real.sum(), inputs=decode_latents(vae,x2_t), create_graph=True, allow_unused=True
            #                 )[0]
            #         if grad1_real is not None:
            #             grad1_penalty = (grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            #         else:
            #             grad1_penalty = 0

            #         if grad2_real is not None:
            #             grad2_penalty = (grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            #         else:
            #             grad2_penalty = 0
            #         if grad1_real is not None or grad2_real is not None:
            #             grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
            #             grad_penalty.backward()

            # ### Noise Sampling ###
            # latent_z1 = torch.randn(batch_size, nz, device=device)
            # latent_z2 = torch.randn(batch_size, nz, device=device)
            
            
            # print("latent_z1:",latent_z1.shape, flush=True)
            
            # ### Predictions through 1to2 & 2to1 Generators for unpaired sample ###
            # # syn
            # x1_0_predict = gen_diffusive_1(torch.cat((fixed_x1_tp1.detach(),l2),axis=1), fixed_t1, latent_z1)
            # x2_0_predict = gen_diffusive_2(torch.cat((fixed_x2_tp1.detach(),l1),axis=1), fixed_t2, latent_z2)
            
            
            # print("x1_0_predict:",x1_0_predict.shape, flush=True)
            
            # # single channel image to discriminators
            # output1 = disc_diffusive_1(decode_latents(vae,x1_0_predict[:, :4, :])).view(-1)
            # output2 = disc_diffusive_2(decode_latents(vae,x2_0_predict[:, :4, :])).view(-1)

            # ### calculates fake error in Discriminator ###
            # errD1_fake = F.softplus(output1)
            # errD2_fake = F.softplus(output2)
            # errD_fake = errD1_fake.mean() + errD2_fake.mean()

            # # If you want to combine errD_real.backward() & errD_fake.b ackward() to errD.backward(), just do that for better simple code ! ! !
            # errD = errD_real + errD_fake

            # ### conditional ###
            # c_output1 = conditional_disc_diffusive_1(decode_latents(vae,x1_0_predict[:, :4, :])).view(-1)
            # c_output2 = conditional_disc_diffusive_2(decode_latents(vae,x2_0_predict[:, :4, :])).view(-1)

            # ### calculates fake error in Discriminator ###
            # err_cD1_fake = F.softplus(c_output1)
            # err_cD2_fake = F.softplus(c_output2)
            # err_cD_fake = err_cD1_fake.mean() + err_cD2_fake.mean()

            # err_cD = err_cD_real + err_cD_fake

            # err_Disc = errD + err_cD
            # err_Disc.backward()

            # ### Update D ###
            # optimizer_disc_diffusive_1.step()
            # optimizer_disc_diffusive_2.step()
            # optimizer_conditional_disc_diffusive_1.step()
            # optimizer_conditional_disc_diffusive_2.step()

            # +------------------------+
            # |   Generator Training   |
            # +------------------------+
            for p in disc_diffusive_1.parameters():
                p.requires_grad = False
            for p in disc_diffusive_2.parameters():
                p.requires_grad = False
            for p in conditional_disc_diffusive_1.parameters():
                p.requires_grad = False
            for p in conditional_disc_diffusive_2.parameters():
                p.requires_grad = False

            gen_diffusive_1.zero_grad()
            gen_diffusive_2.zero_grad()

            ### Sample t ###
            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)

            ### encoding ###
            # 🔥 VAE encoder는 gradient 유지해야 함
            with torch.no_grad():  # VAE는 freeze하고 싶다면
                l1 = encode_images(vae, real_data1)
                l2 = encode_images(vae, real_data2)

            ### Sampling x_t & x_{t+1} ###
            x1_t, x1_tp1 = q_sample_pairs(coeff, l1, t1)
            x2_t, x2_tp1 = q_sample_pairs(coeff, l2, t2)

            print_gpu_memory("Train 2")
            ### Noise Sampling ###
            latent_z1 = torch.randn(batch_size, nz, device=device)
            latent_z2 = torch.randn(batch_size, nz, device=device)

            ### Translation Networks' Prediction ###
            x1_0_predict = gen_diffusive_1(torch.cat((x1_tp1.detach(), l2), axis=1), t1, latent_z1)
            x2_0_predict = gen_diffusive_2(torch.cat((x2_tp1.detach(), l1), axis=1), t2, latent_z2)

            print_gpu_memory("Train 3")
            ### cycle ###
            x1_0_predict_cycle = gen_diffusive_1(torch.cat((x1_tp1.detach(), x2_0_predict[:,:4,:]), axis=1), t1, latent_z1)
            x2_0_predict_cycle = gen_diffusive_2(torch.cat((x2_tp1.detach(), x1_0_predict[:,:4,:]), axis=1), t2, latent_z2)

            x1_pred_latent = x1_0_predict[:, :4, :, :]
            x2_pred_latent = x2_0_predict[:, :4, :, :]
            print_gpu_memory("Train 4")
            # 🔥 VAE decoder도 gradient 유지 필요
            with torch.no_grad():
                x1_decoded = decode_latents(vae, x1_pred_latent)  
                x2_decoded = decode_latents(vae, x2_pred_latent)
            print_gpu_memory("Train 5")
            x1_pred_cycle_latent = x1_0_predict_cycle[:, :4, :, :]
            x2_pred_cycle_latent = x2_0_predict_cycle[:, :4, :, :]

            with torch.no_grad():
                x1_cycle_decoded = decode_latents(vae, x1_pred_cycle_latent)
                x2_cycle_decoded = decode_latents(vae, x2_pred_cycle_latent)
            
            print_gpu_memory("Train 6")
            ### Wavelet Loss ###
            wavelet_loss_syn_value = wavelet_loss([real_data1, real_data2], [x1_decoded, x2_decoded])
            wavelet_loss_cyc_value = wavelet_loss([real_data1, real_data2], [x1_cycle_decoded, x2_cycle_decoded])

            wavelet_loss_syn_values.append(wavelet_loss_syn_value.item())  
            wavelet_loss_cyc_values.append(wavelet_loss_cyc_value.item())

            lambda_cGAN, lambda_cycleGAN = calculate_lambdas(
                wavelet_loss_syn_value.item(),
                wavelet_loss_cyc_value.item()
            )
            print(f'lambda_cGAN: {lambda_cGAN:.4f}, lambda_cycleGAN: {lambda_cycleGAN:.4f}', flush=True)
            lambda_cGAN_values.append(lambda_cGAN)
            lambda_cycleGAN_values.append(lambda_cycleGAN)
            
            ### Discriminator outputs ###
            print_gpu_memory("after wavelet")
            output1 = disc_diffusive_1(x1_decoded).view(-1)
            output2 = disc_diffusive_2(x2_decoded).view(-1)  # 🔥 수정

            errG1 = F.softplus(-output1).mean()
            errG2 = F.softplus(-output2).mean()
            err_cycleG_adv = errG1 + errG2

            ### conditional ###
            c_output1 = conditional_disc_diffusive_1(x1_decoded).view(-1)
            c_output2 = conditional_disc_diffusive_2(x2_decoded).view(-1)

            err_cG1 = F.softplus(-c_output1).mean()
            err_cG2 = F.softplus(-c_output2).mean()
            err_cG_adv = err_cG1 + err_cG2

            ### cycle ###
            cycle_c_output1 = disc_diffusive_1(x1_cycle_decoded).view(-1)
            cycle_c_output2 = disc_diffusive_2(x2_cycle_decoded).view(-1)

            err_cycle_cG1 = F.softplus(-cycle_c_output1).mean()
            err_cycle_cG2 = F.softplus(-cycle_c_output2).mean()
            err_cycle_cG_adv = err_cycle_cG1 + err_cycle_cG2

            ### L1 Losses ###
            err_cG_L1 = F.l1_loss(x1_decoded, real_data1) + F.l1_loss(x2_decoded, real_data2)
            err_cycleG_L1 = F.l1_loss(x1_cycle_decoded, real_data1) + F.l1_loss(x2_cycle_decoded, real_data2)

            ### Total Generator Loss ###
            errG = lambda_cGAN * (err_cG_adv + err_cG_L1) \
                + lambda_cycleGAN * (err_cycleG_adv + err_cycleG_L1) \
                + err_cycle_cG_adv

            # 🔥 디버깅: backward 전에 확인
            print(f"errG.requires_grad: {errG.requires_grad}")
            print(f"errG.grad_fn: {errG.grad_fn}")

            errG.backward()

            optimizer_gen_diffusive_1.step()
            optimizer_gen_diffusive_2.step()

            # +---------------------+
            # |   Display the Loss  |
            # +---------------------+
            global_step += 1
            if iteration % 100 == 0 and rank == 0:
                print(f"Epoch: {epoch}, Iteration: {iteration}")
                print('  [Generator]')
                print(f"  cGAN Loss: {err_cG_adv.item():.4f}")
                print(f"  CycleGAN Loss: {err_cycleG_adv.item():.4f}")
                print(f"  cGAN L1 Loss: {err_cG_L1.item():.4f}")
                print(f"  CycleGAN L1 Loss: {err_cycleG_L1.item():.4f}")
                print(f"  Conbined Cyclic Loss: {errG.item():.4f}")
                print('  [Discriminator]')
                print(f"  cGAN Discriminator Loss: {err_cD.item():.4f}")
                print(f"  CycleGAN Discriminator Loss: {errD.item():.4f}")
                print("-" * 50)
                # ✅ TensorBoard에 Loss 기록
                writer.add_scalar('Loss/Generator/cGAN_adv', err_cG_adv.item(), global_step)
                writer.add_scalar('Loss/Generator/CycleGAN_adv', err_cycleG_adv.item(), global_step)
                writer.add_scalar('Loss/Generator/cGAN_L1', err_cG_L1.item(), global_step)
                writer.add_scalar('Loss/Generator/CycleGAN_L1', err_cycleG_L1.item(), global_step)
                writer.add_scalar('Loss/Generator/Total', errG.item(), global_step)
                writer.add_scalar('Loss/Generator/Cycle_adv', err_cycle_cG_adv.item(), global_step)
                
                writer.add_scalar('Loss/Discriminator/cGAN', err_cD.item(), global_step)
                writer.add_scalar('Loss/Discriminator/CycleGAN', errD.item(), global_step)
                writer.add_scalar('Loss/Discriminator/Total', err_Disc.item(), global_step)
                
                # Wavelet Loss
                writer.add_scalar('Loss/Wavelet/Synthesis', wavelet_loss_syn_value.item(), global_step)
                writer.add_scalar('Loss/Wavelet/Cycle', wavelet_loss_cyc_value.item(), global_step)
                
                # Lambda values
                writer.add_scalar('Hyperparameters/lambda_cGAN', lambda_cGAN, global_step)
                writer.add_scalar('Hyperparameters/lambda_cycleGAN', lambda_cycleGAN, global_step)
                
                # Learning rates
                writer.add_scalar('LearningRate/Generator', optimizer_gen_diffusive_1.param_groups[0]['lr'], global_step)
                writer.add_scalar('LearningRate/Discriminator', optimizer_disc_diffusive_1.param_groups[0]['lr'], global_step)

        if not args.no_lr_decay:

            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()
            scheduler_disc_diffusive_1.step()
            scheduler_disc_diffusive_2.step()
            scheduler_conditional_disc_diffusive_1.step()
            scheduler_conditional_disc_diffusive_2.step()
        
        if rank == 0:
            with torch.no_grad():
                # synthesis
                x1_t = torch.cat((torch.randn_like(l1),l2),axis=1)
                fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)
                x2_t = torch.cat((torch.randn_like(l2),l1),axis=1)
                fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)

                fake_sample1_visual = torch.cat((real_data2, decode_latents(vae,fake_sample1), real_data1),axis=-1)
                torchvision.utils.save_image(fake_sample1_visual, os.path.join(exp_path, 'T1_2_CT_inference_{}.png'.format(epoch)), normalize=True)

                fake_sample2_visual = torch.cat((real_data1, decode_latents(vae,fake_sample2), real_data2),axis=-1)
                torchvision.utils.save_image(fake_sample2_visual, os.path.join(exp_path, 'CT_2_T1_inference_{}.png'.format(epoch)), normalize=True)

                 # ✅ TensorBoard에 이미지 기록
                writer.add_image('Images/T1_to_CT', fake_sample1_visual, epoch)
                writer.add_image('Images/CT_to_T1', fake_sample2_visual, epoch)
                
                # cycle
                cyc_x1_t = torch.cat((torch.randn_like(l1),fake_sample2),axis=1)
                cyc_fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, cyc_x1_t, T, args)

                cyc_x2_t = torch.cat((torch.randn_like(l2),fake_sample1),axis=1)
                cyc_fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, cyc_x2_t, T, args)

                cyc_fake_sample1_visual = torch.cat((real_data2, decode_latents(vae,fake_sample1), cyc_fake_sample2, real_data1),axis=-1)
                torchvision.utils.save_image(cyc_fake_sample1_visual, os.path.join(exp_path, 'T1_recon_{}.png'.format(epoch)), normalize=True)

                cyc_fake_sample2_visual = torch.cat((real_data1, decode_latents(vae,fake_sample2), cyc_fake_sample1, real_data2),axis=-1)
                torchvision.utils.save_image(cyc_fake_sample2_visual, os.path.join(exp_path, 'CT_recon_{}.png'.format(epoch)), normalize=True)
                
                # ✅ Cycle 이미지도 기록
                writer.add_image('Images/T1_reconstruction', cyc_fake_sample1_visual, epoch)
                writer.add_image('Images/CT_reconstruction', cyc_fake_sample2_visual, epoch)
                
                # ✅ Epoch별 평균 loss 기록
                writer.add_scalar('Loss_Epoch/Wavelet_Synthesis_Mean', np.mean(wavelet_loss_syn_values), epoch)
                writer.add_scalar('Loss_Epoch/Wavelet_Cycle_Mean', np.mean(wavelet_loss_cyc_values), epoch)
                writer.add_scalar('Loss_Epoch/lambda_cGAN_Mean', np.mean(lambda_cGAN_values), epoch)
                writer.add_scalar('Loss_Epoch/lambda_cycleGAN_Mean', np.mean(lambda_cycleGAN_values), epoch)
                


                plot_and_save(epoch, wavelet_loss_syn_values, wavelet_loss_cyc_values, lambda_cGAN_values, lambda_cycleGAN_values, plot_save_path)

                if args.save_content:
                    if epoch % args.save_content_every == 0:
                        print('Saving content.')
                        content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                                'gen_diffusive_1_dict': gen_diffusive_1.state_dict(), 'optimizer_gen_diffusive_1': optimizer_gen_diffusive_1.state_dict(),
                                'gen_diffusive_2_dict': gen_diffusive_2.state_dict(), 'optimizer_gen_diffusive_2': optimizer_gen_diffusive_2.state_dict(),
                                'scheduler_gen_diffusive_1': scheduler_gen_diffusive_1.state_dict(), 'disc_diffusive_1_dict': disc_diffusive_1.state_dict(),
                                'scheduler_gen_diffusive_2': scheduler_gen_diffusive_2.state_dict(), 'disc_diffusive_2_dict': disc_diffusive_2.state_dict(),
                                'optimizer_disc_diffusive_1': optimizer_disc_diffusive_1.state_dict(), 'scheduler_disc_diffusive_1': scheduler_disc_diffusive_1.state_dict(),
                                'optimizer_disc_diffusive_2': optimizer_disc_diffusive_2.state_dict(), 'scheduler_disc_diffusive_2': scheduler_disc_diffusive_2.state_dict(),
                                'optimizer_conditional_disc_diffusive_1': optimizer_conditional_disc_diffusive_1.state_dict(), 'scheduler_conditional_disc_diffusive_1': scheduler_conditional_disc_diffusive_1.state_dict(),
                                'optimizer_conditional_disc_diffusive_2': optimizer_conditional_disc_diffusive_2.state_dict(), 'scheduler_conditional_disc_diffusive_2': scheduler_conditional_disc_diffusive_2.state_dict(),
                                'conditional_disc_diffusive_1_dict': conditional_disc_diffusive_1.state_dict(), 'conditional_disc_diffusive_2_dict': conditional_disc_diffusive_2.state_dict(),
                                }
                        
                        torch.save(content, os.path.join(exp_path, 'content.pth'))
                    
                if epoch % args.save_ckpt_every == 0:
                    if args.use_ema:
                        optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                        optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                    torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, 'gen_diffusive_1_{}.pth'.format(epoch)))
                    torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, 'gen_diffusive_2_{}.pth'.format(epoch)))
                    if args.use_ema:
                        optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                        optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
        
    if rank == 0:
        writer.close()
        print("TensorBoard logging finished!")

        # for iteration, (x_val , y_val) in enumerate(val_data_loader):
        
        #     real_data = x_val.to(device, non_blocking=True)
        #     source_data = y_val.to(device, non_blocking=True)
            
        #     x1_t = torch.cat((torch.randn_like(real_data),source_data),axis=1)
        #     #diffusion steps
        #     fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)
        #     fake_sample1 = to_range_0_1(fake_sample1) ; fake_sample1 = fake_sample1/fake_sample1.mean()
        #     real_data = to_range_0_1(real_data) ; real_data = real_data/real_data.mean()

        #     fake_sample1=fake_sample1.cpu().numpy()
        #     real_data=real_data.cpu().numpy()

        #     val_l1_loss[0,epoch,iteration]=abs(fake_sample1 -real_data).mean()
            
        #     val_psnr_values[0,epoch, iteration] = psnr(real_data,fake_sample1, data_range=real_data.max())
        #     #save image
        #     fake_img = (fake_sample1[0, 0] * 255).astype(np.uint8)  # 첫 채널
        #     fake_img_pil = Image.fromarray(fake_img)
        #     fake_img_pil.save(os.path.join(save_dir, f"epoch{epoch}_iter{iteration}_fake.png"))
    
        # for iteration, (y_val , x_val) in enumerate(val_data_loader): 
        
        #     real_data = x_val.to(device, non_blocking=True)
        #     source_data = y_val.to(device, non_blocking=True)
            
        #     x1_t = torch.cat((torch.randn_like(real_data),source_data),axis=1)
        #     #diffusion steps
        #     fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x1_t, T, args)

        #     fake_sample1 = to_range_0_1(fake_sample1) ; fake_sample1 = fake_sample1/fake_sample1.mean()
        #     real_data = to_range_0_1(real_data) ; real_data = real_data/real_data.mean()
            
        #     fake_sample1=fake_sample1.cpu().numpy()
        #     real_data=real_data.cpu().numpy()
        #     val_l1_loss[1,epoch,iteration]=abs(fake_sample1 -real_data).mean()
            
        #     val_psnr_values[1,epoch, iteration] = psnr(real_data,fake_sample1, data_range=real_data.max())

        #     fake_img = (fake_sample1[0, 0] * 255).astype(np.uint8)  # 첫 채널
        #     fake_img_pil = Image.fromarray(fake_img)
        #     fake_img_pil.save(os.path.join(save_dir, f"epoch{epoch}_iter{iteration}_fake.png"))
    

        # print('CT2T1 Generator :',np.nanmean(val_psnr_values[0,epoch,:]))
        # print('T12CT Generator :',np.nanmean(val_psnr_values[1,epoch,:]))

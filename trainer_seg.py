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
from dataset_seg import CreateDatasetSynthesis
import pywt
from utils.utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def combine_channels_to_grayscale(tensor):
    organ_values = [63, 126, 189, 252]
    grayscale = torch.zeros(tensor.shape[0], 1, tensor.shape[2], tensor.shape[3], device=tensor.device)

    for i in range(4):
        mask = (tensor[:, i, :, :] > 0.5)
        grayscale[mask.unsqueeze(1)] = organ_values[i]

    return grayscale

def save_mask_as_png(mask_tensor, file_path):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    mask_image = Image.fromarray(mask_np, mode='L')
    mask_image.save(file_path)

def dice_coefficient_classwise(pred, target, num_classes, threshold=0.5):
    pred = torch.argmax(pred, dim=1)
    target = target.squeeze(1)
    
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))

        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice.mean().item())

    return dice_scores

def iou_classwise(pred, target, num_classes, threshold=0.5):
    """
    클래스별 IoU를 계산합니다.
    pred: [batch, num_classes, height, width]
    target: [batch, 1, height, width]
    num_classes: 총 클래스 수
    """
    pred = torch.argmax(pred, dim=1)  # 예측에서 가장 높은 확률을 가진 클래스 선택
    target = target.squeeze(1)  # [batch, height, width] 형태로 변경

    iou_scores = []
    for cls in range(num_classes):
        # 해당 클래스의 마스크 생성
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        # IoU 계산
        intersection = (pred_cls * target_cls).sum(dim=(1, 2))  # 배치 차원 유지
        union = (pred_cls + target_cls).sum(dim=(1, 2)) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_scores.append(iou.mean().item())  # 배치별 평균을 계산

    return iou_scores

def precision_recall_f1_classwise(pred, target, num_classes):
    """
    클래스별 Precision, Recall, F1 점수를 계산합니다.
    pred: [batch, num_classes, height, width]
    target: [batch, 1, height, width]
    num_classes: 총 클래스 수
    """
    pred = torch.argmax(pred, dim=1)  # 예측에서 가장 높은 확률을 가진 클래스 선택
    target = target.squeeze(1)  # [batch, height, width] 형태로 변경

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for cls in range(num_classes):
        # 해당 클래스의 마스크 생성
        pred_cls = (pred == cls).cpu().numpy().flatten()
        target_cls = (target == cls).cpu().numpy().flatten()

        # Precision, Recall, F1 계산
        precision = precision_score(target_cls, pred_cls, zero_division=0)
        recall = recall_score(target_cls, pred_cls, zero_division=0)
        f1 = f1_score(target_cls, pred_cls, zero_division=0)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return precision_scores, recall_scores, f1_scores
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, smooth=1.0, class_weights=[0.2, 1.0, 1.0, 1.0]):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.class_weights = class_weights  # 클래스 가중치를 설정

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        
        if targets.dim() == 3:
            targets = F.one_hot(targets.long(), num_classes=inputs.shape[1])
            targets = targets.permute(0, 3, 1, 2).float()

        intersection = (inputs * targets).sum(dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth)
        dice_loss = 1 - dice_score.mean()

        targets_ce = targets.argmax(dim=1)

        class_weights = torch.tensor(self.class_weights).to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets_ce, weight=class_weights)

        loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * ce_loss
        return loss


def dice_coefficient(pred, target):
    smooth = 1e-6
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def iou(pred, target):
    smooth = 1e-6
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

from sklearn.metrics import precision_score, recall_score, f1_score

def precision_recall_f1(pred, target):
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    if len(pred_flat) != len(target_flat):
        raise ValueError(f"Inconsistent number of samples: {len(pred_flat)} vs {len(target_flat)}")

    # precision, recall, f1
    precision = precision_score(target_flat.cpu().numpy(), pred_flat.cpu().numpy(), average='macro', zero_division=0)
    recall = recall_score(target_flat.cpu().numpy(), pred_flat.cpu().numpy(), average='macro', zero_division=0)
    f1 = f1_score(target_flat.cpu().numpy(), pred_flat.cpu().numpy(), average='macro', zero_division=0)

    return precision, recall, f1
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.squeeze(1)
        logpt = -F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)
        F_loss = -self.alpha * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

### ### ### ### ### Train Adversarial Diffusion Model ### ### ### ### ###
def train_syndiff(rank, gpu, args):
    from backbones.discriminator_noiseless import Discriminator_small, Discriminator_large
    
    from backbones.ncsnpp_generator_adagn import NCSNpp
    
    import backbones.generator_resnet 

    from utils.EMA import EMA

    from seg_models.FCNGCN import FCNGCN
    
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
    train_dataset = CreateDatasetSynthesis(phase = "train", input_path = args.input_path, contrast1 = args.contrast1, contrast2 = args.contrast2)
    val_dataset = CreateDatasetSynthesis(phase = "val", input_path = args.input_path, contrast1 = args.contrast1, contrast2 = args.contrast2)
    
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
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)

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

    # Segmentation
    seg_input_channels = 1
    seg_classes = 4

    net_seg = FCNGCN(seg_input_channels, seg_classes).to(device)
    # criterion = nn.BCEWithLogitsLoss()
    # criterion_seg = FocalLoss()
    criterion_seg = CombinedLoss()

    optimizer_seg = optim.Adam(net_seg.parameters(), lr=1e-3, betas=(args.beta1, args.beta2))
    scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, args.num_epoch, eta_min=1e-5)

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

    exp = args.exp
    output_path = args.output_path

    exp_path = os.path.join(output_path,exp)
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

    ### ### ### --- Fine-Tunning for Segmentation --- ### ### ###
    if args.seg_fine_tunning:
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
    

    # +------------------+
    # |  Start Training  |
    # +------------------+
    for epoch in range(init_epoch, args.num_epoch):
        
        wavelet_loss_syn_values = []
        wavelet_loss_cyc_values = []
        lambda_cGAN_values = []
        lambda_cycleGAN_values = []

        dice_scores_real = []
        dice_scores_fake = []
        iou_scores_real = []
        iou_scores_fake = []
        precision_scores_real = []
        recall_scores_real = []
        f1_scores_real = []
        precision_scores_fake = []
        recall_scores_fake = []
        f1_scores_fake = []

        ### Sampling different subsets of the dataset in each epoch ###
        train_sampler.set_epoch(epoch)
        for iteration, (x1, x2, x2_mask) in enumerate(train_data_loader):
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

            ### sample x_t and x_{t+1} ###
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x1_t.requires_grad = True
            
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)
            x2_t.requires_grad = True
            # +--------------------------+
            # |  Discriminator Training  |
            # +--------------------------+
            ### training discriminator with real sample ###
            D1_real = disc_diffusive_1(real_data1).view(-1)
            D2_real = disc_diffusive_2(real_data2).view(-1)
            
            cD1_real = conditional_disc_diffusive_1(real_data1).view(-1)
            cD2_real = conditional_disc_diffusive_2(real_data2).view(-1)

            ### calculates real error in Discriminator ###
            
            # Evaluates how well the discriminator distinguishes "real data" : F.softplus(-D{1,2}_real)
            errD1_real = F.softplus(-D1_real)
            errD1_real = errD1_real.mean()
            
            errD2_real = F.softplus(-D2_real)
            errD2_real = errD2_real.mean()
            errD_real = errD1_real + errD2_real

            ### conditional ###
            err_cD1_real = F.softplus(-cD1_real)
            err_cD1_real = err_cD1_real.mean()
            
            err_cD2_real = F.softplus(-cD2_real)
            err_cD2_real = err_cD2_real.mean()
            err_cD_real = err_cD1_real + err_cD2_real

            ### apply the gradient penalty in each epoch ###
            if args.lazy_reg is None:
                grad1_real = torch.autograd.grad(
                            outputs=D1_real.sum(), inputs=x1_t, create_graph=True
                            )[0]
                grad1_penalty = (
                                grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                grad2_real = torch.autograd.grad(
                            outputs=D2_real.sum(), inputs=x2_t, create_graph=True
                            )[0]
                grad2_penalty = (
                                grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                
                grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad1_real = torch.autograd.grad(
                            outputs=D1_real.sum(), inputs=x1_t, create_graph=True, allow_unused=True
                            )[0]
                    grad2_real = torch.autograd.grad(
                            outputs=D2_real.sum(), inputs=x2_t, create_graph=True, allow_unused=True
                            )[0]
                    if grad1_real is not None:
                        grad1_penalty = (grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    else:
                        grad1_penalty = 0

                    if grad2_real is not None:
                        grad2_penalty = (grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    else:
                        grad2_penalty = 0
                    if grad1_real is not None or grad2_real is not None:
                        grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
                        grad_penalty.backward()

            ### Noise Sampling ###
            latent_z1 = torch.randn(batch_size, nz, device=device)
            latent_z2 = torch.randn(batch_size, nz, device=device)
            
            ### Predictions through 1to2 & 2to1 Generators for unpaired sample ###
            # syn
            x1_0_predict = gen_diffusive_1(torch.cat((x1_tp1.detach(),real_data2),axis=1), t1, latent_z1)
            x2_0_predict = gen_diffusive_2(torch.cat((x2_tp1.detach(),real_data1),axis=1), t2, latent_z2)
            
            # single channel image to discriminators
            output1 = disc_diffusive_1(x1_0_predict[:, [0], :]).view(-1)
            output2 = disc_diffusive_2(x2_0_predict[:, [0], :]).view(-1)

            ### calculates fake error in Discriminator ###
            errD1_fake = F.softplus(output1)
            errD2_fake = F.softplus(output2)
            errD_fake = errD1_fake.mean() + errD2_fake.mean()

            # If you want to combine errD_real.backward() & errD_fake.backward() to errD.backward(), just do that for better simple code ! ! !
            errD = errD_real + errD_fake

            ### conditional ###
            c_output1 = conditional_disc_diffusive_1(x1_0_predict[:, [0], :]).view(-1)
            c_output2 = conditional_disc_diffusive_2(x2_0_predict[:, [0], :]).view(-1)

            ### calculates fake error in Discriminator ###
            err_cD1_fake = F.softplus(c_output1)
            err_cD2_fake = F.softplus(c_output2)
            err_cD_fake = err_cD1_fake.mean() + err_cD2_fake.mean()

            err_cD = err_cD_real + err_cD_fake

            err_Disc = errD + err_cD
            err_Disc.backward()

            ### Update D ###
            optimizer_disc_diffusive_1.step()
            optimizer_disc_diffusive_2.step()
            optimizer_conditional_disc_diffusive_1.step()
            optimizer_conditional_disc_diffusive_2.step()

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
            
            ### Sampling x_t & x_{t+1} ###
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)
            
            ### Noise Sampling ###
            latent_z1 = torch.randn(batch_size, nz,device=device)
            latent_z2 = torch.randn(batch_size, nz,device=device)

            ### Translation Networks' Prediction ###
            # syn
            x1_0_predict = gen_diffusive_1(torch.cat((x1_tp1.detach(),real_data2),axis=1), t1, latent_z1)
            x2_0_predict = gen_diffusive_2(torch.cat((x2_tp1.detach(),real_data1),axis=1), t2, latent_z2)

            # cycle
            x1_0_predict_cycle = gen_diffusive_1(torch.cat((x1_tp1.detach(), x2_0_predict[:,[0],:]),axis=1), t1, latent_z1)
            x2_0_predict_cycle = gen_diffusive_2(torch.cat((x2_tp1.detach(), x1_0_predict[:,[0],:]),axis=1), t2, latent_z2)

            ### Wavelet Loss ###
            real_images_syn = [real_data1, real_data2]
            generated_images_syn = [x1_0_predict[:,[0],:], x2_0_predict[:,[0],:]]
            wavelet_loss_syn_value = wavelet_loss(real_images_syn, generated_images_syn)

            real_images_cyc = [real_data1, real_data2]
            generated_images_cyc = [x1_0_predict_cycle[:,[0],:], x2_0_predict_cycle[:,[0],:]]
            wavelet_loss_cyc_value = wavelet_loss(real_images_cyc, generated_images_cyc)

            wavelet_loss_syn_values.append(wavelet_loss_syn_value.item())
            wavelet_loss_cyc_values.append(wavelet_loss_cyc_value.item())

            lambda_cGAN, lambda_cycleGAN = calculate_lambdas(
                wavelet_loss_syn_value.item(),
                wavelet_loss_cyc_value.item()
            )
            print('lambda_cGAN, lambda_cycleGAN :', lambda_cGAN, lambda_cycleGAN)
            lambda_cGAN_values.append(lambda_cGAN)
            lambda_cycleGAN_values.append(lambda_cycleGAN)
            save_high_freq_components(epoch, real_images_syn, generated_images_syn)

            ### D output for fake sample ###
            output1 = disc_diffusive_1(x1_0_predict[:,[0],:]).view(-1)
            output2 = disc_diffusive_2(x2_0_predict[:,[0],:]).view(-1)

            # Evaluates how well the fake data created by the generator is judged to be 'real' by the discriminator : - output{1,2}
            errG1 = F.softplus(-output1)
            errG1 = errG1.mean()

            errG2 = F.softplus(-output2)
            errG2 = errG2.mean()
            
            ### Diffusive Generator ###
            err_cycleG_adv = errG1 + errG2

            ### conditional ###
            c_output1 = conditional_disc_diffusive_1(x1_0_predict[:,[0],:]).view(-1)
            c_output2 = conditional_disc_diffusive_2(x2_0_predict[:,[0],:]).view(-1)

            err_cG1 = F.softplus(-c_output1)
            err_cG1 = err_cG1.mean()

            err_cG2 = F.softplus(-c_output2)
            err_cG2 = err_cG2.mean()
            
            err_cG_adv = err_cG1 + err_cG2

            ### Cycle Loss for real & Diffusive module ###
            err_cG1_L1 = F.l1_loss(x1_0_predict[:,[0],:],real_data1)
            err_cG2_L1 = F.l1_loss(x2_0_predict[:,[0],:],real_data2)
            err_cG_L1 = err_cG1_L1 + err_cG2_L1

            ### Cycle Loss for real & Non-Diffusive module ###
            err_cycleG1_L1 = F.l1_loss(x1_0_predict_cycle[:,[0],:], real_data1)
            err_cycleG2_L1 = F.l1_loss(x2_0_predict_cycle[:,[0],:], real_data2)
            err_cycleG_L1 = err_cycleG1_L1 + err_cycleG2_L1

            torch.autograd.set_detect_anomaly(True)

            # Segmentation - seg_fine_tunning
            # x2_mask = x2_mask.to(device, non_blocking=True).float().unsqueeze(1)

            # seg_real_t2_pred = net_seg(real_data2)
            # seg_fake_t2_pred = net_seg(x2_0_predict_cycle[:,[0],:])

            # seg_real_t2_loss = criterion_seg(seg_real_t2_pred, x2_mask)
            # seg_fake_t2_loss = criterion_seg(seg_fake_t2_pred, x2_mask)

            # seg_loss = seg_real_t2_loss + seg_fake_t2_loss
            # seg_loss *= 10.0 # weighted
            # optimizer_seg.zero_grad()

            errG = lambda_cycleGAN * (err_cG_adv + err_cG_L1) \
                 + lambda_cGAN * (err_cycleG_adv + err_cycleG_L1) \
                #  + seg_loss

            errG.backward()

            ### Update Non-diffusive Generator & Diffusive Generator ###
            optimizer_gen_diffusive_1.step()
            optimizer_gen_diffusive_2.step()
            optimizer_seg.step()

            # Seg Eval.
            # dice_real = dice_coefficient_classwise(seg_real_t2_pred, x2_mask, seg_classes)
            # dice_fake = dice_coefficient_classwise(seg_fake_t2_pred, x2_mask, seg_classes)

            # iou_real = iou_classwise(seg_real_t2_pred, x2_mask, seg_classes)
            # iou_fake = iou_classwise(seg_fake_t2_pred, x2_mask, seg_classes)

            # precision_real, recall_real, f1_real = precision_recall_f1_classwise(seg_real_t2_pred, x2_mask, seg_classes)
            # precision_fake, recall_fake, f1_fake = precision_recall_f1_classwise(seg_fake_t2_pred, x2_mask, seg_classes)

            # x_predict_train_visual = torch.cat((x1_0_predict[:, [0], :], x1_0_predict_cycle[:,[0],:], x1_tp1, x2_0_predict[:, [0], :], x2_0_predict_cycle[:,[0],:], x2_tp1),axis=-1)
            # x_predict_train_visual = (x_predict_train_visual + 1) / 2.0
            # torchvision.utils.save_image(x_predict_train_visual, os.path.join(exp_path, 'x_predict_train_{}.png'.format(epoch)), normalize=False)


            # seg_real_train_gray = combine_channels_to_grayscale(seg_real_t2_pred)
            # seg_fake_train_gray = combine_channels_to_grayscale(seg_fake_t2_pred)
            # # print(seg_real_train_gray.shape, seg_fake_train_gray.shape, x2_mask.shape)
            # x_predict_train_visual_seg = torch.cat((seg_real_train_gray, seg_fake_train_gray, x2_mask),axis=-1)
            # torchvision.utils.save_image(x_predict_train_visual_seg, os.path.join(exp_path, 'x_predict_train_seg_{}.png'.format(epoch)), normalize=False)

            # for i in range(4):
            #     print(f"Class {i} - Dice Real: {dice_real[i]:.4f}, Dice Fake: {dice_fake[i]:.4f}")
            #     print(f"Class {i} - IoU Real: {iou_real[i]:.4f}, IoU Fake: {iou_fake[i]:.4f}")
            #     print(f"Class {i} - Precision Real: {precision_real[i]:.4f}, Recall Real: {recall_real[i]:.4f}, F1 Real: {f1_real[i]:.4f}")
            #     print(f"Class {i} - Precision Fake: {precision_fake[i]:.4f}, Recall Fake: {recall_fake[i]:.4f}, F1 Fake: {f1_fake[i]:.4f}")

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

        if not args.no_lr_decay:

            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()
            scheduler_disc_diffusive_1.step()
            scheduler_disc_diffusive_2.step()
            scheduler_conditional_disc_diffusive_1.step()
            scheduler_conditional_disc_diffusive_2.step()

            scheduler_seg.step()

        if rank == 0:
            # synthesis
            x1_t = torch.cat((torch.randn_like(real_data1),real_data2),axis=1)
            fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)
            x2_t = torch.cat((torch.randn_like(real_data2),real_data1),axis=1)
            fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)

            fake_sample1_visual = torch.cat((real_data2, fake_sample1, real_data1),axis=-1)
            torchvision.utils.save_image(fake_sample1_visual, os.path.join(exp_path, 'T1_2_CT_inference_{}.png'.format(epoch)), normalize=True)

            fake_sample2_visual = torch.cat((real_data1, fake_sample2, real_data2),axis=-1)
            torchvision.utils.save_image(fake_sample2_visual, os.path.join(exp_path, 'CT_2_T1_inference_{}.png'.format(epoch)), normalize=True)            

            # cycle
            cyc_x1_t = torch.cat((torch.randn_like(real_data1),fake_sample2),axis=1)
            cyc_fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, cyc_x1_t, T, args)

            cyc_x2_t = torch.cat((torch.randn_like(real_data2),fake_sample1),axis=1)
            cyc_fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, cyc_x2_t, T, args)

            # Segmentation
            x2_mask = x2_mask.to(device, non_blocking=True).float()
            if x2_mask.dim() == 3:
                x2_mask = x2_mask.unsqueeze(0)  # Add batch dimension

            seg_real_result = net_seg(real_data2)
            seg_fake_result = net_seg(cyc_fake_sample2)
            
            # print(seg_real_result.shape, seg_fake_result.shape, x2_mask.shape)
            seg_real_gray = combine_channels_to_grayscale(seg_real_result)
            seg_fake_gray = combine_channels_to_grayscale(seg_fake_result)
            seg_result = torch.cat((seg_real_gray, seg_fake_gray, x2_mask),axis=-1)
            # seg_result *= 255
            # torchvision.utils.save_image(seg_result, os.path.join(exp_path, 'seg_result_epoch_{}.png'.format(epoch)), normalize=False)
            save_mask_as_png(seg_result, os.path.join(exp_path, 'seg_result_epoch_{}.png'.format(epoch)))
            cyc_fake_sample1_visual = torch.cat((real_data2, fake_sample1, cyc_fake_sample2, real_data1),axis=-1)
            torchvision.utils.save_image(cyc_fake_sample1_visual, os.path.join(exp_path, 'T1_recon_{}.png'.format(epoch)), normalize=True)

            cyc_fake_sample2_visual = torch.cat((real_data1, fake_sample2, cyc_fake_sample1, real_data2),axis=-1)
            torchvision.utils.save_image(cyc_fake_sample2_visual, os.path.join(exp_path, 'CT_recon_{}.png'.format(epoch)), normalize=True)
            
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
                               'conditional_disc_diffusive_1_dict': conditional_disc_diffusive_1.state_dict(), 'conditional_disc_diffusive_2_dict': conditional_disc_diffusive_2.state_dict(), 'seg_model_dict': net_seg.state_dict(), 'optimizer_seg':optimizer_seg.state_dict(), 'scheduler_seg': scheduler_seg.state_dict()
                               }
                    
                    torch.save(content, os.path.join(exp_path, 'content.pth'))
                
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                    # optimizer_seg.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, 'gen_diffusive_1_{}.pth'.format(epoch)))
                torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, 'gen_diffusive_2_{}.pth'.format(epoch)))
                torch.save(net_seg.state_dict(), os.path.join(exp_path, 'net_seg_{}.pth'.format(epoch)))

                if args.use_ema:
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                    # optimizer_seg.swap_parameters_with_ema(store_params_in_ema=True)

        # ONLY E2E
        for iteration, (x_val, y_val, y_mask) in enumerate(val_data_loader):

            real_data = x_val.to(device, non_blocking=True)
            source_data = y_val.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True).float().unsqueeze(1)

            # Generate fake_sample1
            x1_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
            fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)
            fake_sample1 = to_range_0_1(fake_sample1) 
            fake_sample1 = fake_sample1 / fake_sample1.mean()
            real_data = to_range_0_1(real_data) 
            real_data = real_data / real_data.mean()

            # Calculate L1 loss for fake_sample1 and real_data
            fake_sample1_np = fake_sample1.cpu().numpy()
            real_data_np = real_data.cpu().numpy()
            val_l1_loss[0, epoch, iteration] = abs(fake_sample1_np - real_data_np).mean()
            
            val_psnr_values[0, epoch, iteration] = psnr(real_data_np, fake_sample1_np, data_range=real_data_np.max())

            # Generate fake_sample2
            x2_t = torch.cat((torch.randn_like(source_data), fake_sample1), axis=1)
            fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)
            fake_sample2 = to_range_0_1(fake_sample2)
            fake_sample2 = fake_sample2 / fake_sample2.mean()
            source_data = to_range_0_1(source_data) 
            source_data = source_data / source_data.mean()

            # Convert to numpy arrays
            fake_sample2_np = fake_sample2.cpu().numpy()
            source_data_np = source_data.cpu().numpy()

            # Calculate L1 loss for fake_sample2 and source_data
            val_l1_loss[1, epoch, iteration] = abs(fake_sample2_np - source_data_np).mean()
            
            val_psnr_values[1, epoch, iteration] = psnr(source_data_np, fake_sample2_np, data_range=source_data_np.max())

            # Convert back to Tensors for segmentation
            fake_sample1 = torch.from_numpy(fake_sample1_np).float().to(device)
            fake_sample2 = torch.from_numpy(fake_sample2_np).float().to(device)
            source_data = torch.from_numpy(source_data_np).float().to(device)

            # Segmentation
            # seg_real_val = net_seg(source_data)
            # seg_fake_val = net_seg(fake_sample2)
            
            # Metrics
            # dice_real = dice_coefficient(seg_real_val, y_mask)
            # dice_fake = dice_coefficient(seg_fake_val, y_mask)
            # iou_real = iou(seg_real_val, y_mask)
            # iou_fake = iou(seg_fake_val, y_mask)
            # precision_real, recall_real, f1_real = precision_recall_f1(seg_real_val, y_mask)
            # precision_fake, recall_fake, f1_fake = precision_recall_f1(seg_fake_val, y_mask)

            # Append metrics
            # dice_scores_real.append(dice_real)
            # dice_scores_fake.append(dice_fake)
            # iou_scores_real.append(iou_real)
            # iou_scores_fake.append(iou_fake)
            # precision_scores_real.append(precision_real)
            # recall_scores_real.append(recall_real)
            # f1_scores_real.append(f1_real)
            # precision_scores_fake.append(precision_fake)
            # recall_scores_fake.append(recall_fake)
            # f1_scores_fake.append(f1_fake)

        print('CT2T1 Generator :',np.nanmean(val_psnr_values[0,epoch,:]))
        print('T12CT Cyclic Generator :',np.nanmean(val_psnr_values[1,epoch,:]))

        # print(f"Average Dice Real: {np.mean(dice_scores_real):.4f}, Average Dice Fake: {np.mean(dice_scores_fake):.4f}")
        # print(f"Average IoU Real: {np.mean(iou_scores_real):.4f}, Average IoU Fake: {np.mean(iou_scores_fake):.4f}")
        # print(f"Average Precision Real: {np.mean(precision_scores_real):.4f}, Average Recall Real: {np.mean(recall_scores_real):.4f}, Average F1 Real: {np.mean(f1_scores_real):.4f}")
        # print(f"Average Precision Fake: {np.mean(precision_scores_fake):.4f}, Average Recall Fake: {np.mean(recall_scores_fake):.4f}, Average F1 Fake: {np.mean(f1_scores_fake):.4f}")

# brain
# nohup python train_seg.py --image_size 256 --exp exp_syndiff --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --contrast1 CT --contrast2 T2 --num_epoch 2000 --ngf 32 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-5 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 2 --save_content --local_rank 0 --input_path ../datasets/new_brain/train --output_path ./output_brain/for/results_minmax0525_cross --port_num 5009 --seg_fine_tunning > brain_e2e.log 2>&1 &

# kumc duct
# nohup python train_seg.py --image_size 512 --exp exp_syndiff --num_channels 2 --num_channels_dae 32 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 1 --batch_size 1 --contrast1 CT --contrast2 T1 --num_epoch 2000 --ngf 32 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 2 --save_content --local_rank 0 --input_path ../datasets/kumc_invisible/train --output_path ./output_kumc_duct/for/results_minmax0525 --port_num 5000 --resume > kumc_duct_minmax0525.log 2>&1 &

# kumc multi
# nohup python train_seg.py --image_size 512 --exp exp_syndiff --num_channels 2 --num_channels_dae 32 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 1 --batch_size 1 --contrast1 CT --contrast2 T1 --num_epoch 2000 --ngf 32 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 2 --save_content --local_rank 0 --input_path ../datasets/kumc_visible/train --output_path ./output_kumc_mlti/for/results_minmax0525 --port_num 5001 --resume > kumc_multi_minmax0525.log 2>&1 &

# chaos
# nohup python train_seg.py --image_size 256 --exp exp_syndiff --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 1 --batch_size 1 --contrast1 CT --contrast2 T1 --num_epoch 2000 --ngf 32 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 2 --save_content --local_rank 0 --input_path ../datasets/CHAOS/train --output_path ./output_chaos/for/results_minmax0525 --port_num 5002 > chaos_minmax0525.log 2>&1 &



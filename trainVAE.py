import torch 
import os
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class MRIDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.images = []

        patient_dirs = sorted(os.listdir(directory))

        for patient_dir in patient_dirs:
            if patient_dir.endswith(".txt"):
                continue
            patient_dir_path = os.path.join(directory, patient_dir)
            image_files = sorted(os.listdir(patient_dir_path))
            for image_file in image_files:
                if image_file.endswith(".png"):
                    image_path = os.path.join(patient_dir_path, image_file)
                    image = Image.open(image_path).convert("L")
                    self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

class CTDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.images = []

        patient_dirs = sorted(os.listdir(directory))

        for patient_dir in patient_dirs:
            if patient_dir.endswith(".txt"):
                continue
            patient_dir_path = os.path.join(directory, patient_dir)
            image_files = sorted(os.listdir(patient_dir_path))
            for image_file in image_files:
                if image_file.endswith(".png"):
                    image_path = os.path.join(patient_dir_path, image_file)
                    image = Image.open(image_path).convert("L") # to Gray Scale
                    self.images.append(image)  # No cropping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

def CreateDatasetSynthesis(phase, input_path, contrast1='CT', contrast2='T1'):
    assert phase in ['train', 'val', 'test'], "Phase should be 'train', 'val', or 'test'"

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    ct_target_file = os.path.join(input_path, contrast1)
    data_fs_ct = CTDataset(ct_target_file, transform=transform)

    mrt1_target_file = os.path.join(input_path, contrast2)
    data_fs_mrt1 = MRIDataset(mrt1_target_file, transform=transform)

    if phase in ['train', 'val']:
        total_samples = len(data_fs_ct)
        num_train = int(total_samples * 0.80)
        num_val = total_samples - num_train

        indices = list(range(total_samples))
        train_indices, val_indices = train_test_split(indices, train_size=num_train, test_size=num_val, shuffle=True, random_state=42)

        if phase == 'train':
            selected_indices = train_indices
        elif phase == 'val':
            selected_indices = val_indices

        data_fs_ct = torch.stack([data_fs_ct[i] for i in selected_indices])
        data_fs_mrt1 = torch.stack([data_fs_mrt1[i] for i in selected_indices])

    elif phase == 'test':
        selected_indices = list(range(len(data_fs_ct)))
        data_fs_ct = torch.stack([data_fs_ct[i] for i in selected_indices])
        data_fs_mrt1 = torch.stack([data_fs_mrt1[i] for i in selected_indices])

    dataset = torch.utils.data.TensorDataset(data_fs_ct, data_fs_mrt1)

    return dataset


device = torch.device('cuda:{}'.format(0))
batch_size = 1
input_path = "../datasets/kumc_invisible/train"
cache_dir = "./dataset_cache"
os.makedirs(cache_dir, exist_ok=True)
contrast1 , contrast2 = 'CT' , 'T1'
train_cache_file = os.path.join(cache_dir, f"train_{contrast1}_{contrast2}.pkl")
val_cache_file = os.path.join(cache_dir, f"val_{contrast1}_{contrast2}.pkl")

if not os.path.exists(train_cache_file):
    train_dataset = CreateDatasetSynthesis(phase="train", input_path=input_path, contrast1= contrast1, contrast2 = contrast2)
    with open(train_cache_file, 'wb') as f:
        pickle.dump(train_dataset, f)

if not os.path.exists(val_cache_file):
    val_dataset = CreateDatasetSynthesis(phase="val", input_path=input_path, contrast1=contrast1, contrast2=contrast2)
    with open(val_cache_file, 'wb') as f:
        pickle.dump(val_dataset, f)

with open(train_cache_file, 'rb') as f:
    train_dataset = pickle.load(f)
with open(val_cache_file, 'rb') as f:
    val_dataset = pickle.load(f)
num_process_per_node = 1
num_proc_node = 1 
world_size = num_proc_node * num_process_per_node
rank = 0
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=world_size,rank=rank)
train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True,sampler=train_sampler,drop_last = True)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,num_replicas=world_size,rank=rank)
val_data_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True,sampler=val_sampler,drop_last = True)


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
from tqdm import tqdm

class AutoencoderKL_Gray(nn.Module):
    def __init__(self, pretrained_model_name="stabilityai/sd-vae-ft-mse"):
        super().__init__()
        # 원래 AutoencoderKL 구조 로드
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name)

        # 3채널 → 1채널에 맞게 첫/마지막 conv 수정
        old_first = self.vae.encoder.conv_in
        self.vae.encoder.conv_in = nn.Conv2d(
            1, old_first.out_channels, kernel_size=3, stride=1, padding=1
        )

        old_last = self.vae.decoder.conv_out
        self.vae.decoder.conv_out = nn.Conv2d(
            old_last.in_channels, 1, kernel_size=3, stride=1, padding=1
        )

    def encode(self, x):
        h = self.vae.encoder(x)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # KL divergence를 위한 샘플링
        std = torch.exp(0.5 * logvar)
        z = mean + torch.randn_like(mean) * std
        return z, mean, logvar

    def decode(self, z):
        dec = self.vae.post_quant_conv(z)
        x_recon = self.vae.decoder(dec)
        return x_recon

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mean, logvar

# -----------------------------
# 2. VAE 초기화 및 GPU 이동
# -----------------------------
vae = AutoencoderKL_Gray().cuda()

# -----------------------------
# 3. Loss 함수 정의
# -----------------------------
def vae_loss(recon, original, mean, logvar, kl_weight=0.00001):
    """
    VAE loss = Reconstruction loss + KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon, original, reduction='mean')
    
    # KL divergence loss
    # KL(N(μ, σ²) || N(0, 1)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss

# -----------------------------
# 4. Optimizer 설정
# -----------------------------
num_epochs = 10
optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=0.01)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs * len(train_data_loader) * 2  # CT + MRI 둘 다 학습
)

# -----------------------------
# 5. 시각화 함수
# -----------------------------
def show_recon(original, recon, step, epoch, modality):
    """학습 중 재구성 결과 시각화"""
    # Denormalize: [-1, 1] -> [0, 1]
    original = (original * 0.5 + 0.5).clamp(0, 1)
    recon = (recon * 0.5 + 0.5).clamp(0, 1)
    
    # Create grid
    grid = make_grid(torch.cat([original, recon], dim=0), nrow=original.size(0))
    
    plt.figure(figsize=(12, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis("off")
    plt.title(f"Epoch {epoch} | Step {step} | {modality}: Original (top) / Reconstructed (bottom)")
    plt.tight_layout()
    plt.show()
    plt.close()

# -----------------------------
# 6. 학습 루프 (CT + MRI 둘 다)
# -----------------------------
log_interval = 100
kl_weight = 0.00001  # KL divergence weight

vae.train()

for epoch in range(num_epochs):
    epoch_ct_loss = 0
    epoch_mri_loss = 0
    epoch_ct_recon = 0
    epoch_mri_recon = 0
    epoch_ct_kl = 0
    epoch_mri_kl = 0
    
    pbar = tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for step, (ct, mri) in enumerate(pbar):
        ct = ct.to("cuda")
        mri = mri.to("cuda")

        # 1채널 확인 및 normalization to [-1, 1]
        if ct.shape[1] != 1:
            ct = ct.mean(dim=1, keepdim=True)
        if mri.shape[1] != 1:
            mri = mri.mean(dim=1, keepdim=True)
            
        # 이미 normalize되어 있으면 skip (범위 확인)
        if ct.min() >= 0 and ct.max() <= 1:
            ct = ct * 2.0 - 1.0
        if mri.min() >= 0 and mri.max() <= 1:
            mri = mri * 2.0 - 1.0

        # ========== CT 학습 ==========
        recon_ct, mean_ct, logvar_ct = vae(ct)
        loss_ct, recon_loss_ct, kl_loss_ct = vae_loss(
            recon_ct, ct, mean_ct, logvar_ct, kl_weight=kl_weight
        )

        optimizer.zero_grad()
        loss_ct.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_ct_loss += loss_ct.item()
        epoch_ct_recon += recon_loss_ct.item()
        epoch_ct_kl += kl_loss_ct.item()

        # ========== MRI 학습 ==========
        recon_mri, mean_mri, logvar_mri = vae(mri)
        loss_mri, recon_loss_mri, kl_loss_mri = vae_loss(
            recon_mri, mri, mean_mri, logvar_mri, kl_weight=kl_weight
        )

        optimizer.zero_grad()
        loss_mri.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_mri_loss += loss_mri.item()
        epoch_mri_recon += recon_loss_mri.item()
        epoch_mri_kl += kl_loss_mri.item()

        # Progress bar 업데이트
        pbar.set_postfix({
            'CT_loss': f'{loss_ct.item():.4f}',
            'MRI_loss': f'{loss_mri.item():.4f}',
            'CT_recon': f'{recon_loss_ct.item():.4f}',
            'MRI_recon': f'{recon_loss_mri.item():.4f}'
        })

        # Visualization
        if step % log_interval == 0:
            with torch.no_grad():
                show_recon(ct[:4], recon_ct[:4], step, epoch+1, "CT")
                show_recon(mri[:4], recon_mri[:4], step, epoch+1, "MRI")

    # Epoch summary
    n = len(train_data_loader)
    avg_ct_loss = epoch_ct_loss / n
    avg_mri_loss = epoch_mri_loss / n
    avg_ct_recon = epoch_ct_recon / n
    avg_mri_recon = epoch_mri_recon / n
    avg_ct_kl = epoch_ct_kl / n
    avg_mri_kl = epoch_mri_kl / n
    
    print(f"\n{'='*70}")
    print(f"[Epoch {epoch+1}] Summary:")
    print(f"  CT  → Loss: {avg_ct_loss:.4f} | Recon: {avg_ct_recon:.4f} | KL: {avg_ct_kl:.6f}")
    print(f"  MRI → Loss: {avg_mri_loss:.4f} | Recon: {avg_mri_recon:.4f} | KL: {avg_mri_kl:.6f}")
    print(f"{'='*70}\n")

# -----------------------------
# 7. 모델 저장
# -----------------------------
torch.save({
    'epoch': num_epochs,
    'model_state_dict': vae.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'vae_ct_mri_shared.pt')

print("✅ Fine-tuning 완료! 모델이 'vae_ct_mri_shared.pt'로 저장되었습니다.")
print("🔬 이 VAE는 CT와 MRI 둘 다 인코딩/디코딩할 수 있습니다!")


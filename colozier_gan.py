
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms
from torchvision.utils import save_image

from skimage import color
from tqdm import tqdm
import gradio as gr


#* U-NET GENERATOR ARCHITECTURE

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNetGenerator, self).__init__()

        # Encoder (Downsampling)
        self.down1 = UNetDown(in_channels, 64, normalize=False)  # 128x128
        self.down2 = UNetDown(64, 128)  # 64x64
        self.down3 = UNetDown(128, 256)  # 32x32
        self.down4 = UNetDown(256, 512, dropout=0.5)  # 16x16
        self.down5 = UNetDown(512, 512, dropout=0.5)  # 8x8
        self.down6 = UNetDown(512, 512, dropout=0.5)  # 4x4
        self.down7 = UNetDown(512, 512, dropout=0.5)  # 2x2
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)  # 1x1

        # Decoder (Upsampling with skip connections)
        self.up1 = UNetUp(512, 512, dropout=0.5)  # 2x2
        self.up2 = UNetUp(1024, 512, dropout=0.5)  # 4x4
        self.up3 = UNetUp(1024, 512, dropout=0.5)  # 8x8
        self.up4 = UNetUp(1024, 512, dropout=0.5)  # 16x16
        self.up5 = UNetUp(1024, 256)  # 32x32
        self.up6 = UNetUp(512, 128)  # 64x64
        self.up7 = UNetUp(256, 64)  # 128x128

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


# PATCHGAN DISCRIMINATOR ARCHITECTURE

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),  # 128x128
            *discriminator_block(64, 128),  # 64x64
            *discriminator_block(128, 256),  # 32x32
            *discriminator_block(256, 512),  # 16x16
            nn.Conv2d(512, 1, 4, padding=1)  # 16×16 patch output
        )

    def forward(self, img_L, img_ab):
        # Concatenate L channel with ab channels
        img_input = torch.cat((img_L, img_ab), 1)
        return self.model(img_input)


#  WEIGHTS INITIALIZATION

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


#  DATASET CLASS (LAB COLOR SPACE)

class LandscapeColorizationDataset(Dataset):
    def __init__(self, image_dir, img_size=256, subset=None):
        self.img_size = img_size
        self.image_paths = []
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
        
        # Optional: use subset of images
        if subset is not None:
            self.image_paths = self.image_paths[:subset]
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        
        # Transforms for resizing
        self.transform = transforms.Resize((img_size, img_size), Image.BICUBIC)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img = self.transform(img)
            
            # Convert PIL to numpy
            img_np = np.array(img)
            
            # Convert RGB to LAB color space
            img_lab = color.rgb2lab(img_np).astype(np.float32)
            
            # Extract L channel (0-100 range) and ab channels (-128 to 127 range)
            L = img_lab[:, :, 0]
            ab = img_lab[:, :, 1:]
            
            # Normalize to [-1, 1] range
            L = (L / 50.0) - 1.0  # L: 0-100 -> -1 to 1
            ab = ab / 128.0  # ab: -128 to 127 -> -1 to 1
            
            # Convert to tensors [C, H, W]
            L = torch.from_numpy(L).unsqueeze(0)  # [1, 256, 256]
            ab = torch.from_numpy(ab.transpose(2, 0, 1))  # [2, 256, 256]
            
            return L, ab
            
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return random tensor on error
            return torch.randn(1, self.img_size, self.img_size), \
                   torch.randn(2, self.img_size, self.img_size)


# TRAINING FUNCTION (MIXED PRECISION)

def train_pix2pix(
    train_dir,
    num_epochs=100,
    batch_size=32,
    lr=0.0002,
    beta1=0.5,
    beta2=0.999,
    lambda_L1=100,
    save_interval=10,
    checkpoint_dir='checkpoints'
):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    generator = UNetGenerator(in_channels=1, out_channels=2).to(device)
    discriminator = PatchDiscriminator(in_channels=3).to(device)
    
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    criterion_GAN = nn.MSELoss()  # LSGAN loss
    criterion_L1 = nn.L1Loss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    scaler_G = GradScaler(enabled=(device.type == 'cuda'))
    scaler_D = GradScaler(enabled=(device.type == 'cuda'))
    
    train_dataset = LandscapeColorizationDataset(train_dir, img_size=256)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nTraining Configuration:")
    print(f"- Epochs: {num_epochs}")
    print(f"- Batch Size: {batch_size}")
    print(f"- Training Images: {len(train_dataset)}")
    print(f"- Learning Rate: {lr}")
    print(f"- Lambda L1: {lambda_L1}")
    print(f"- Mixed Precision: {device.type == 'cuda'}")
    print()
    
    # Training loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (real_L, real_ab) in enumerate(pbar):
            real_L = real_L.to(device)
            real_ab = real_ab.to(device)
            
            batch_size_current = real_L.size(0)
            
            # =======================
            # Train Discriminator
            # =======================
            optimizer_D.zero_grad()
            
            with autocast(enabled=(device.type == 'cuda')):
                # Generate fake ab channels
                fake_ab = generator(real_L)
                
                # Real loss
                pred_real = discriminator(real_L, real_ab)
                target_real = torch.ones_like(pred_real, device=device)
                loss_real = criterion_GAN(pred_real, target_real)
                
                # Fake loss
                pred_fake = discriminator(real_L, fake_ab.detach())
                target_fake = torch.zeros_like(pred_fake, device=device)
                loss_fake = criterion_GAN(pred_fake, target_fake)
                
                # Total discriminator loss
                loss_D = 0.5 * (loss_real + loss_fake)
            
            # Backward pass with gradient scaling
            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            
            # =======================
            # Train Generator
            # =======================
            optimizer_G.zero_grad()
            
            with autocast(enabled=(device.type == 'cuda')):
                # Generate fake ab channels
                fake_ab = generator(real_L)
                
                # Adversarial loss
                pred_fake = discriminator(real_L, fake_ab)
                target_real = torch.ones_like(pred_fake, device=device)
                loss_GAN = criterion_GAN(pred_fake, target_real)
                
                # L1 loss (pixel-wise reconstruction)
                loss_L1 = criterion_L1(fake_ab, real_ab)
                
                # Total generator loss
                loss_G = loss_GAN + lambda_L1 * loss_L1
            
            # Backward pass with gradient scaling
            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            
            # Accumulate losses
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}',
                'L1': f'{loss_L1.item():.4f}'
            })
        
        # Epoch statistics
        avg_loss_G = epoch_loss_G / len(train_loader)
        avg_loss_D = epoch_loss_D / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - G_loss: {avg_loss_G:.4f}, D_loss: {avg_loss_D:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % save_interval == 0:
            checkpoint_path_G = os.path.join(checkpoint_dir, f'generator_epoch_{epoch+1}.pth')
            checkpoint_path_D = os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch+1}.pth')
            
            torch.save(generator.state_dict(), checkpoint_path_G)
            torch.save(discriminator.state_dict(), checkpoint_path_D)
            print(f"Saved checkpoints at epoch {epoch+1}")
    
    # Save final models
    torch.save(generator.state_dict(), os.path.join(checkpoint_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'discriminator_final.pth'))
    print("\nTraining completed!")


# GRADCAM 

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, channel_idx=0):
        """
        Generate attention map for specific output channel
        channel_idx: 0 for 'a' channel, 1 for 'b' channel
        """
        self.model.eval()
        
        # Enable gradient computation for input
        input_image = input_image.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_image)
        
        # Zero gradients
        self.model.zero_grad()
        if input_image.grad is not None:
            input_image.grad.zero_()
        
        # Backward pass for specific channel
        target = output[:, channel_idx, :, :]
        target.backward(torch.ones_like(target), retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # ReLU to keep only positive influences
        cam = torch.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def visualize_attention(self, input_image, channel_name='a'):
        channel_idx = 0 if channel_name == 'a' else 1
        
        # Generate CAM (needs gradients enabled)
        cam = self.generate_cam(input_image, channel_idx)
        
        # Convert to numpy
        cam_np = cam[0, 0].detach().cpu().numpy()
        
        # Resize to match input size
        cam_resized = np.array(Image.fromarray(cam_np).resize((256, 256), Image.BICUBIC))
        
        # Apply colormap (jet: red=high attention, blue=low attention)
        heatmap = cm.jet(cam_resized)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Convert input grayscale to RGB for overlay
        input_np = input_image[0, 0].cpu().numpy()
        input_np = ((input_np + 1.0) * 50.0).clip(0, 100)  # Denormalize L channel
        input_rgb = np.stack([input_np] * 3, axis=-1)
        input_rgb = (input_rgb * 255 / 100).astype(np.uint8)
        
        # Blend: 60% heatmap + 40% original
        overlay = (0.6 * heatmap + 0.4 * input_rgb).astype(np.uint8)
        
        return Image.fromarray(overlay)


# GRADIO INTERFACE

def lab_to_rgb(L, ab):
    
    # Denormalize
    L = ((L + 1.0) * 50.0).clamp(0, 100)  # [-1, 1] -> [0, 100]
    ab = (ab * 128.0).clamp(-128, 127)  # [-1, 1] -> [-128, 127]
    
    # Combine L and ab
    lab = torch.cat([L, ab], dim=1)  # [B, 3, H, W]
    
    # Convert to numpy [H, W, 3]
    lab_np = lab[0].permute(1, 2, 0).cpu().numpy()
    
    # LAB to RGB
    rgb_np = color.lab2rgb(lab_np)
    rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(rgb_np)


def colorize_image(generator, gradcam_a, gradcam_b, input_image_pil, device):
    """
    Colorize grayscale image and generate attention maps
    Returns: (colorized_image, attention_map_a, attention_map_b)
    """
    # Preprocess input
    img_np = np.array(input_image_pil.resize((256, 256), Image.BICUBIC))
    
    # Convert to LAB
    if len(img_np.shape) == 2:  # Already grayscale
        L = img_np.astype(np.float32)
    else:  # RGB image
        img_lab = color.rgb2lab(img_np).astype(np.float32)
        L = img_lab[:, :, 0]
    
    # Normalize L channel
    L = (L / 50.0) - 1.0  # [0, 100] -> [-1, 1]
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 256, 256]
    
    # Generate colorization (without gradients for efficiency)
    generator.eval()
    with torch.no_grad():
        fake_ab = generator(L_tensor)
    
    # Convert to RGB
    colorized_rgb = lab_to_rgb(L_tensor, fake_ab)
    
    # Generate attention maps (needs separate forward pass with gradients)
    # Create a new tensor for GradCAM (allows gradients)
    L_tensor_gradcam = L_tensor.clone().detach().requires_grad_(True)
    attention_a = gradcam_a.visualize_attention(L_tensor_gradcam, channel_name='a')
    attention_b = gradcam_b.visualize_attention(L_tensor_gradcam, channel_name='b')
    
    return colorized_rgb, attention_a, attention_b


def create_gradio_interface(generator_path, device):
    # Load generator
    generator = UNetGenerator(in_channels=1, out_channels=2).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    
    # Initialize GradCAM for both channels
    # Target layer: last upsampling layer (up7)
    gradcam_a = GradCAM(generator, target_layer=generator.up7.model[0])
    gradcam_b = GradCAM(generator, target_layer=generator.up7.model[0])
    
    def inference_fn(input_image):
        """Gradio inference function"""
        if input_image is None:
            return None, None, None
        
        colorized, attn_a, attn_b = colorize_image(
            generator, gradcam_a, gradcam_b, input_image, device
        )
        return colorized, attn_a, attn_b
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=inference_fn,
        inputs=gr.Image(type="pil", label="Upload Grayscale Image"),
        outputs=[
            gr.Image(type="pil", label="Colorized Image"),
            gr.Image(type="pil", label="Attention Map (Red-Green Channel)"),
            gr.Image(type="pil", label="Attention Map (Blue-Yellow Channel)")
        ],
        title="Image Colorization with Explainable AI",
        description="""
        ### Automatic Image Colorization using Pix2Pix GAN
        
        **How it works:**
        1. Upload a grayscale (black & white) image
        2. The model predicts realistic colors
        3. View two attention maps showing which image regions the model focused on:
           - **Red-Green Channel**: Shows areas important for red/green color prediction
           - **Blue-Yellow Channel**: Shows areas important for blue/yellow color prediction
        
        **Attention Maps Legend:**
        -  Red regions = High attention (model focuses here)
        -  Blue regions = Low attention (less important for color prediction)
        
        The GradCAM attention maps provide explainability without affecting model performance.
        """,
        examples=None,
        allow_flagging="never"
    )
    
    return interface


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # =========================================================================
    # TRAINING (UNCOMMENT TO TRAIN)
    # =========================================================================
    """
    # Training configuration
    TRAIN_DIR = 'GAN_color_images'  # Relative path - works on any system
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0002
    LAMBDA_L1 = 100
    CHECKPOINT_DIR = 'checkpoints'
    
    # Start training
    train_pix2pix(
        train_dir=TRAIN_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        lambda_L1=LAMBDA_L1,
        save_interval=10,
        checkpoint_dir=CHECKPOINT_DIR
    )
    """
    
    # =========================================================================
    # INFERENCE - GRADIO INTERFACE
    # =========================================================================
    
    # Path to trained generator (change to your checkpoint)
    GENERATOR_PATH = 'checkpoints/generator_final.pth'
    
    # Check if model exists
    if not os.path.exists(GENERATOR_PATH):
        print(f"ERROR: Model checkpoint not found at {GENERATOR_PATH}")
        print("Please train the model first or update the GENERATOR_PATH variable.")
        print("\nTo train:")
        print("1. Uncomment the training section above")
        print("2. Run this script")
        print("3. Wait for training to complete (~100 epochs)")
        exit(1)
    
    # Create and launch Gradio interface
    print("Loading model and creating Gradio interface...")
    interface = create_gradio_interface(GENERATOR_PATH, device)
    
    print("\nLaunching Gradio interface...")
    print("You can:")
    print("1. Use it locally in your browser")
    print("2. Share it publicly via the generated link")
    
    interface.launch(share=True)

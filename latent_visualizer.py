import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from config import CONFIG
from utils import get_latent_data

# 1. SCATTER PLOTS (Encoder View)

def plot_scatter(vae, title_suffix):
    print(f"Generating Scatter Plots for {title_suffix}...")

    # Get data
    dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
    z, y = get_latent_data(vae, loader)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(z)
    sc1 = axes[0].scatter(z_tsne[:, 0], z_tsne[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
    axes[0].set_title(f"t-SNE: {title_suffix}")
    fig.colorbar(sc1, ax=axes[0], ticks=range(10))

    # UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(random_state=42)
    z_umap = reducer.fit_transform(z)
    sc2 = axes[1].scatter(z_umap[:, 0], z_umap[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
    axes[1].set_title(f"UMAP: {title_suffix}")
    fig.colorbar(sc2, ax=axes[1], ticks=range(10))

    plt.show()

# 2. MANIFOLD TRAVERSAL (Decoder View)

def plot_grid_traversal(vae, title_suffix, target_digit):
    print(f"Generating PCA Grid for {title_suffix} (Digit {target_digit})...")

    # 1. Fit PCA on real Z data to find "active" directions
    dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
    z_real, _ = get_latent_data(vae, loader, num_samples=5000)

    pca = PCA(n_components=2)
    pca.fit(z_real)

    # 2. Create Grid in PCA space (Standard deviations)
    grid_size = 20
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)

    # Flatten grid and project back to 8D space
    flat_grid = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    z_grid = pca.inverse_transform(flat_grid)
    z_grid = torch.from_numpy(z_grid).float().to(CONFIG['device'])

    # 3. Fix the Class Label (Conditioning)
    c = torch.full((len(z_grid),), target_digit).to(CONFIG['device'])
    c_oh = F.one_hot(c, 10).float()

    # 4. Decode
    vae.eval()
    with torch.no_grad():
        imgs = vae.decoder(z_grid, c_oh).view(-1, 1, 28, 28)

    # Plot
    grid_img = make_grid(imgs, nrow=grid_size, padding=0)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title(f"PCA Grid ({title_suffix}) - Class {target_digit}")
    plt.show()

def plot_slerp(vae, title_suffix, target_digit):
    print(f"Generating SLERP for {title_suffix} (Digit {target_digit})...")

    # Mathematical formula for Spherical Interpolation
    def slerp(val, low, high):
        omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
        so = np.sin(omega)
        if so == 0: return (1.0-val) * low + val * high
        return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

    n_steps = 10
    n_rows = 10

    z_all = []

    for _ in range(n_rows):
        # Pick two random points in latent space
        z1 = np.random.randn(CONFIG['z_dim'])
        z2 = np.random.randn(CONFIG['z_dim'])

        # Interpolate between them
        for alpha in np.linspace(0, 1, n_steps):
            z_interp = slerp(alpha, z1, z2)
            z_all.append(z_interp)

    z_tensor = torch.tensor(np.array(z_all)).float().to(CONFIG['device'])

    # Fix Class
    c = torch.full((len(z_tensor),), target_digit).to(CONFIG['device'])
    c_oh = F.one_hot(c, 10).float()

    vae.eval()
    with torch.no_grad():
        imgs = vae.decoder(z_tensor, c_oh).view(-1, 1, 28, 28)

    grid_img = make_grid(imgs, nrow=n_steps, padding=2)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title(f"SLERP ({title_suffix}) - Class {target_digit}")
    plt.show()

# Wrapper to call all plots at once
def generate_all_plots(vae, name, forgot_list=[]):
    plot_scatter(vae, name)

    check_list = forgot_list + [9]

    for digit in check_list:
        plot_grid_traversal(vae, name, digit)
        plot_slerp(vae, name, digit)

# 3. LABEL INTERPOLATION (Morphing)
def plot_morph_digits(vae, title_suffix, start_digit, end_digit, n_steps=12, n_rows=10):
    print(f"Generating Morph from {start_digit} to {end_digit} ({title_suffix})...")

    # Helper: Spherical Linear Interpolation
    def slerp(val, low, high):
        low_norm = low / torch.norm(low, dim=1, keepdim=True)
        high_norm = high / torch.norm(high, dim=1, keepdim=True)
        omega = torch.acos(torch.clamp(torch.sum(low_norm * high_norm, dim=1), -1, 1))
        so = torch.sin(omega)
        res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + \
              (torch.sin(val * omega) / so).unsqueeze(1) * high
        return torch.where(so.unsqueeze(1) == 0, (1.0 - val) * low + val * high, res)

    all_imgs = []

    c_start = F.one_hot(torch.tensor([start_digit]*n_rows), 10).float().to(CONFIG['device'])
    c_end   = F.one_hot(torch.tensor([end_digit]*n_rows), 10).float().to(CONFIG['device'])

    z_start = torch.randn(n_rows, CONFIG['z_dim']).to(CONFIG['device'])
    z_end   = torch.randn(n_rows, CONFIG['z_dim']).to(CONFIG['device'])

    vae.eval()
    with torch.no_grad():
        for alpha in np.linspace(0, 1, n_steps):
            # Interpolate Label (Linear mixing)
            c_interp = (1 - alpha) * c_start + alpha * c_end

            # Interpolate Latent Style (SLERP)
            z_interp = slerp(alpha, z_start, z_end)

            # Decode
            recon = vae.decoder(z_interp, c_interp).view(-1, 1, 28, 28)
            all_imgs.append(recon)

    stacked_imgs = torch.stack(all_imgs, dim=1)
    stacked_imgs = stacked_imgs.view(-1, 1, 28, 28)

    grid_img = make_grid(stacked_imgs, nrow=n_steps, padding=2)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title(f"Morph {start_digit} -> {end_digit} ({title_suffix})")
    plt.show()

def plot_scatter_highlighted(vae, title_suffix):
    print(f"Generating Highlighted Scatter Plots for {title_suffix}...")

    dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
    z, y = get_latent_data(vae, loader, num_samples=3000)

    # Define Colors manually
    colors = []
    sizes = []
    orders = []

    for label in y:
        if label == 0:
            colors.append('red')
            sizes.append(20)
            orders.append(2)
        elif label == 1:
            colors.append('blue')
            sizes.append(20)
            orders.append(2)
        else:
            colors.append('lightgrey')
            sizes.append(5)
            orders.append(1)

    colors = np.array(colors)
    sizes = np.array(sizes)
    orders = np.array(orders)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(z)

    # Plot background (Grey) first, then targets (Red/Blue)
    for order in [1, 2]:
        mask = (orders == order)
        if np.sum(mask) > 0:
            axes[0].scatter(z_tsne[mask, 0], z_tsne[mask, 1],
                           c=colors[mask], s=sizes[mask], alpha=0.6, label=f"Order {order}")

    axes[0].set_title(f"t-SNE: {title_suffix}\n(Red=0, Blue=1, Grey=Others)")

    # UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(random_state=42)
    z_umap = reducer.fit_transform(z)

    for order in [1, 2]:
        mask = (orders == order)
        if np.sum(mask) > 0:
            axes[1].scatter(z_umap[mask, 0], z_umap[mask, 1],
                           c=colors[mask], s=sizes[mask], alpha=0.6)

    axes[1].set_title(f"UMAP: {title_suffix}\n(Red=0, Blue=1, Grey=Others)")

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Digit 0'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Digit 1'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey', label='Others')]
    axes[0].legend(handles=legend_elements)

    plt.show()

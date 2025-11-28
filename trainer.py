import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import copy

from config import CONFIG
from models import OneHotCVAE
from utils import cycle, loss_function


def calculate_fim(vae):
    print("\n[PHASE 3] Calculating Fisher Information Matrix (The Correct Way)...")
    fisher_dict = {}
    for name, param in vae.named_parameters():
        fisher_dict[name] = torch.zeros_like(param.data)


    vae.train()
    optimizer = optim.Adam(vae.parameters()) # Dummy

    n_samples = 50000

    print(f"Accumulating gradients over {n_samples} samples...")
    for _ in tqdm(range(n_samples)):
        # Sample ONE vector
        z = torch.randn(1, CONFIG['z_dim']).to(CONFIG['device'])
        c = torch.randint(0, 10, (1,)).to(CONFIG['device'])
        c_oh = F.one_hot(c, 10).float()

        # Synthetic Data Generation
        with torch.no_grad():
            vae.eval()
            sampled_data = vae.decoder(z, c_oh)

        # Backprop on that ONE sample
        vae.train()
        optimizer.zero_grad()
        recon, mu, log_var = vae(sampled_data, c_oh)

        loss = loss_function(recon, sampled_data, mu, log_var)
        loss.backward()

        # Accumulate Squared Gradients
        for name, param in vae.named_parameters():
            if param.grad is not None:
                # F = E[grad^2]
                fisher_dict[name] += (param.grad.data ** 2) / n_samples

    return fisher_dict


def train_model():
    print(f"\n[PHASE 2] Pre-training (Paper Exact: 100k Steps)...")

    # Paper Architecture (256 -> 512)
    vae = OneHotCVAE(z_dim=CONFIG['z_dim'], h_dim1=256, h_dim2=512).to(CONFIG['device'])
    optimizer = optim.Adam(vae.parameters(), lr=CONFIG['lr']) # 1e-4

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)

    train_iter = cycle(train_loader)

    vae.train()

    n_iters = 100000
    log_freq = 1000
    train_loss = 0
    train_bce = 0
    train_kld = 0

    pbar = tqdm(range(n_iters))

    for step in pbar:
        data, label = next(train_iter)
        data = data.to(CONFIG['device'])
        label_oh = F.one_hot(label, 10).float().to(CONFIG['device'])

        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data, label_oh)

        # Calculate loss
        BCE = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = BCE + KLD

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bce += BCE.item()
        train_kld += KLD.item()

        if (step + 1) % log_freq == 0:
            avg_loss = train_loss / log_freq
            avg_bce = train_bce / log_freq
            avg_kld = train_kld / log_freq

            pbar.set_description(f"Step {step+1} | Loss: {avg_loss:.0f} | BCE: {avg_bce:.0f} | KLD: {avg_kld:.1f}")

            train_loss = 0
            train_bce = 0
            train_kld = 0

    return vae


def train_forget_multiple(vae, fisher_dict, labels_to_drop):
    """
    Args:
        labels_to_drop: List of integers, e.g., [0, 1]
    """
    print(f"\n[PHASE 4] Forgetting Digits {labels_to_drop}...")

    # Reset Hyperparameters
    CONFIG['lmbda'] = 100

    vae_clone = copy.deepcopy(vae)
    vae_clone.eval()
    for p in vae_clone.parameters(): p.requires_grad = False

    params_mle = {n: p.data.clone() for n, p in vae.named_parameters()}

    optimizer = optim.Adam(vae.parameters(), lr=CONFIG['lr'])
    vae.train()

    labels_keep = [i for i in range(10) if i not in labels_to_drop]

    labels_forget = labels_to_drop

    pbar = tqdm(range(CONFIG['forget_steps']))
    for step in pbar:
        optimizer.zero_grad()

        # 1. FORGET (Targets -> Noise)
        # Randomly choose which "forget label" to use for this batch
        idx_f = torch.randint(0, len(labels_forget), (CONFIG['batch_size'],))
        c_forget = torch.tensor([labels_forget[i] for i in idx_f]).to(CONFIG['device'])
        c_forget_oh = F.one_hot(c_forget, 10).float()

        # Target is Random Noise
        x_noise = torch.rand(CONFIG['batch_size'], 1, 28, 28).to(CONFIG['device'])

        recon_f, mu_f, log_var_f = vae(x_noise, c_forget_oh)
        loss_forget = loss_function(recon_f, x_noise, mu_f, log_var_f)

        # 2. REMEMBER (Keep -> Teacher)
        # Randomly choose a "keep label"
        idx_k = torch.randint(0, len(labels_keep), (CONFIG['batch_size'],))
        c_keep = torch.tensor([labels_keep[i] for i in idx_k]).to(CONFIG['device'])
        c_keep_oh = F.one_hot(c_keep, 10).float()

        # Latent code for replay
        z_keep = torch.randn(CONFIG['batch_size'], CONFIG['z_dim']).to(CONFIG['device'])

        with torch.no_grad():
            x_replay = vae_clone.decoder(z_keep, c_keep_oh).view(-1, 1, 28, 28)

        recon_r, mu_r, log_var_r = vae(x_replay, c_keep_oh)
        loss_replay = loss_function(recon_r, x_replay, mu_r, log_var_r)

        # 3. EWC PENALTY
        loss_ewc = 0
        for n, p in vae.named_parameters():
            _loss = fisher_dict[n] * (p - params_mle[n]) ** 2
            loss_ewc += _loss.sum()

        # Total Loss
        loss = loss_forget + (CONFIG['gamma'] * loss_replay) + (CONFIG['lmbda'] * loss_ewc)

        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            pbar.set_postfix({
                'F': f"{loss_forget.item():.1e}",
                'R': f"{loss_replay.item():.1e}",
                'EWC': f"{loss_ewc.item():.1e}"
            })

    return vae


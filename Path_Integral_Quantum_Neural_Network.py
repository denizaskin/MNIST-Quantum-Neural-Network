## Written by Deniz Askin - Helped by ChatGPT-o3
"""
Incremental path‑integral learner:
 * inputs  x_a  = 784 MNIST pixel positions on [0,1]
 * outputs χ_b = 10 learnable class “slit centres” on [0,1]
 * weights  W_ab = √[1/(2π i)] exp[i (x_a−χ_b)^2 / 2]
 * learning: complex Hebbian weight tweak + proportional χ update
No autograd, no gradient descent.
"""

import torch, torchvision, torchvision.transforms as T, torch.nn.functional as F, math

# ---------- hyper‑parameters -------------------------------------------------
BATCH   = 256
EPOCHS  = 200
LR_CHI  = 0.05         # proportional step for χ
SEED    = 42
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

LR_BIAS = 0.10        # step size for complex bias
LR_AMP  = 0.02        # step size for real per‑pixel amplitudes

# ---------- dataset ---------------------------------------------------------
transform = T.ToTensor()          # keep pixel values in [0,1]
train_set = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_set  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=1000, shuffle=False)

# ---------- coordinates and kernel builder ----------------------------------
x_in = torch.linspace(0.0, 1.0, 28*28, device=DEVICE)   # input coords  (784,)
xb   = x_in.unsqueeze(0)                                # (1,784)

# complex prefactor  √(1/2π i) = 1/√(2π) * e^{-iπ/4}
prefactor = (1 / math.sqrt(2*math.pi)) * torch.exp(torch.tensor(-1j*math.pi/4,
                                     dtype=torch.complex64, device=DEVICE))

# learnable class centres χ_b  ∈ [0,1]  (10,)
chi = torch.nn.Parameter(torch.rand(10, device=DEVICE))

# learnable pixel‑wise real amplitudes  amp_{b a}  (10,784)
amp = torch.nn.Parameter(torch.ones(10, 784, device=DEVICE))

# learnable complex bias for each class
bias = torch.nn.Parameter(torch.zeros(10, dtype=torch.complex64, device=DEVICE))

# helper: complex one‑hot targets
def onehot(labels, n=10):
    out = torch.zeros(labels.size(0), n, dtype=torch.complex64, device=DEVICE)
    out[torch.arange(labels.size(0), device=DEVICE), labels] = 1 + 0j
    return out

def kernel(chi_vec: torch.Tensor) -> torch.Tensor:
    """
    Build the (10,784) complex weight matrix: prefactor * exp(i/2 (x_a-χ_b)^2).
    """
    xa = chi_vec.unsqueeze(1)                      # (10,1)
    phase = 0.5 * (xa - xb) ** 2                   # (10,784)
    return amp.to(torch.complex64) * prefactor * torch.exp(1j * phase)  # (10,784)

# ---------- training loop ---------------------------------------------------
for epoch in range(EPOCHS):
    correct_train = total_train = 0
    running_loss = 0.0

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader, 1):
        # complex inputs ----------------------------------------------------
        x_batch = x_batch.reshape(-1, 28*28).to(torch.complex64).to(DEVICE)  # (B,784)
        yt      = onehot(y_batch.to(DEVICE))                                 # (B,10)

        # forward -----------------------------------------------------------
        K = kernel(chi)                                   # (10,784)
        y_complex = x_batch @ K.T.conj() + bias           # (B,10)
        logits    = y_complex.abs() ** 2                  # real, (B,10)

        # mean‑squared error (matches δ used by Hebbian update) ----------
        delta = yt - y_complex                            # complex error
        loss  = (delta.abs() ** 2).mean()
        running_loss += loss.item()

        # ------- algebraic parameter updates (no autograd) -----------------
        with torch.no_grad():
            # bias update (complex Hebbian tweak)
            bias += LR_BIAS * delta.mean(dim=0)

            # amplitude update (Hebbian real part)
            hebb = (delta.conj().unsqueeze(2) * x_batch.unsqueeze(1)).mean(dim=0).real  # (10,784)
            amp += LR_AMP * hebb
            amp.clamp_(0.0)

            # χ update (proportional to magnitude‑squared error)
            x_bar    = x_batch.real.mean(dim=1, keepdim=True)                 # (B,1)
            chi_step = (delta.abs()**2 * (x_bar - chi.unsqueeze(0))).mean(dim=0)
            chi     += LR_CHI * chi_step.real
            chi.clamp_(0.0, 1.0)

        # training accuracy -------------------------------------------------
        preds = logits.argmax(dim=1)
        correct_train += (preds.cpu() == y_batch).sum().item()
        total_train   += y_batch.size(0)

        if batch_idx % 50 == 0:
            avg_loss = running_loss / batch_idx
            print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} "
                  f"Loss {avg_loss:.4f}")
    
    train_acc = correct_train / total_train

# ---------- evaluation ---------------------------------------------------
with torch.no_grad():
    correct_test = total_test = 0
    for x_test, y_test in test_loader:
        x_test = x_test.reshape(-1, 28*28).to(torch.complex64).to(DEVICE)
        y_complex = x_test @ kernel(chi).T.conj() + bias
        logits_test = y_complex.abs() ** 2
        preds_test = logits_test.argmax(dim=1).cpu()
        correct_test += (preds_test == y_test).sum().item()
        total_test += y_test.size(0)

    test_acc = correct_test / total_test

print(f"Epoch {epoch+1}  TrainAcc {train_acc:5.2%}  TestAcc {test_acc:5.2%}")
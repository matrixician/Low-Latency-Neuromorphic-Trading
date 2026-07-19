"""
Recurrent Leaky-Integrate-and-Fire (LIF) spiking network that predicts the
conditional intensities of a marked point process (one intensity per event
type). Trained by binned Poisson negative log-likelihood -- the same objective
class as Hawkes MLE, so the comparison is apples-to-apples.

Causality: the intensity emitted at bin t is read out from the network state
*before* bin t's input events are injected, so lambda_i(t) depends only on
events strictly earlier than t (no look-ahead / leakage).
"""
import numpy as np
import torch
import torch.nn as nn

s
# ---------------------------------------------------------------- surrogate
class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside forward; smooth surrogate gradient on the backward pass."""
    k = 10.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + FastSigmoidSpike.k * x.abs()) ** 2
        return grad_out * sg


spike_fn = FastSigmoidSpike.apply


# --------------------------------------------------------------- the model
class RecurrentLIF(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, dt,
                 tau_min=0.05, tau_max=1.0, tau_out_min=0.05, tau_out_max=2.0,
                 seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n_hidden = n_hidden
        self.dt = dt

        # heterogeneous, fixed membrane time constants -> spread of decays
        taus = torch.linspace(tau_min, tau_max, n_hidden)
        self.register_buffer("mem_decay", torch.exp(-dt / taus))          # (H,)
        # heterogeneous readout-trace time constants (multi-timescale readout)
        tout = torch.linspace(tau_out_min, tau_out_max, n_hidden)
        self.register_buffer("out_decay", torch.exp(-dt / tout))          # (H,)

        self.W_in = nn.Parameter(torch.randn(n_in, n_hidden, generator=g) * 0.5)
        # recurrent weights: small init, zero self-connection bias not enforced
        self.W_rec = nn.Parameter(torch.randn(n_hidden, n_hidden, generator=g)
                                  / np.sqrt(n_hidden))
        self.W_out = nn.Parameter(torch.randn(n_hidden, n_out, generator=g)
                                  / np.sqrt(n_hidden))
        self.b_out = nn.Parameter(torch.zeros(n_out))
        self.thresh = 1.0

    def forward(self, x, state=None, return_hidden=False):
        """x: (B, L, n_in) binned input spikes. Returns lam: (B, L, n_out)."""
        B, L, _ = x.shape
        H = self.n_hidden
        dev = x.device
        if state is None:
            mem = torch.zeros(B, H, device=dev)
            s_prev = torch.zeros(B, H, device=dev)
            r = torch.zeros(B, H, device=dev)          # readout trace
        else:
            mem, s_prev, r = state

        lam_list = []
        hid_list = []
        for t in range(L):
            # ---- read out intensity from state BEFORE injecting x[t] (causal)
            lam_t = torch.nn.functional.softplus(r @ self.W_out + self.b_out)
            lam_list.append(lam_t)
            if return_hidden:
                hid_list.append(r)
            # ---- evolve hidden state with current input
            cur = x[:, t, :] @ self.W_in + s_prev @ self.W_rec
            mem = self.mem_decay * mem * (1.0 - s_prev) + cur       # soft reset
            s = spike_fn(mem - self.thresh)
            r = self.out_decay * r + s
            s_prev = s

        lam = torch.stack(lam_list, dim=1)                          # (B,L,out)
        state = (mem.detach(), s_prev.detach(), r.detach())
        if return_hidden:
            return lam, torch.stack(hid_list, dim=1), state
        return lam, state


# --------------------------------------------------------------- utilities
def bin_events(times, types, dt, T, D):
    """Return integer count grid (n_bins, D) and left-edge grid times."""
    n_bins = int(np.floor(T / dt))
    counts = np.zeros((n_bins, D), dtype=np.float32)
    idx = np.floor(times / dt).astype(int)
    keep = idx < n_bins
    for b, c in zip(idx[keep], types[keep]):
        counts[b, c] += 1.0
    grid = np.arange(n_bins) * dt
    return counts, grid


def poisson_nll_torch(lam, counts, dt):
    """Binned Poisson NLL (drops constant log n!)."""
    rate = torch.clamp(lam * dt, min=1e-8)
    return (rate - counts * torch.log(rate)).sum()


def make_windows(counts, L):
    """Split (n_bins, D) into non-overlapping windows -> (nW, L, D)."""
    n_bins, D = counts.shape
    nW = n_bins // L
    counts = counts[: nW * L]
    return counts.reshape(nW, L, D)


def train_snn(model, counts_train, dt, epochs=40, batch=32, lr=3e-3,
              L=200, weight_decay=1e-5, seed=0, log_every=5):
    """Train by binned Poisson NLL with truncated BPTT over windows.
    Input at window step t predicts intensity at step t (causal via model)."""
    torch.manual_seed(seed)
    D = counts_train.shape[1]
    W = make_windows(counts_train, L)                  # (nW, L, D)
    W = torch.tensor(W, dtype=torch.float32)
    nW = W.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_events = float(counts_train.sum())
    history = []
    for ep in range(epochs):
        perm = torch.randperm(nW)
        tot = 0.0
        for i in range(0, nW, batch):
            idx = perm[i:i + batch]
            xb = W[idx]                                # (b, L, D) input = events
            lam, _ = model(xb)                         # causal intensities
            loss = poisson_nll_torch(lam, xb, dt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        # per-event NLL for readability
        per_ev = float(tot / max(n_events, 1.0))
        history.append(per_ev)
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  train NLL/event = {per_ev:.4f}")
    return history


@torch.no_grad()
def predict_intensity(model, counts, dt, L=200):
    """Run the model over a long sequence in chunks, carrying state across
    chunk boundaries, returning intensity aligned to each bin (n_bins, D)."""
    D = counts.shape[1]
    n_bins = counts.shape[0]
    x = torch.tensor(counts, dtype=torch.float32).unsqueeze(0)  # (1, n_bins, D)
    state = None
    out = []
    for start in range(0, n_bins, L):
        chunk = x[:, start:start + L, :]
        lam, state = model(chunk, state=state)
        out.append(lam.squeeze(0))
    return torch.cat(out, dim=0).numpy()

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class S4D(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.H = d_model
        self.N = d_state

        A_real = -0.5 * torch.ones(d_state)
        A_imag = math.pi * torch.arange(d_state).float()
        self._A = nn.Parameter(torch.view_as_real(torch.complex(A_real, A_imag)))

        self.B = nn.Parameter(torch.randn(d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))
        log_dt = (
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.log_dt = nn.Parameter(log_dt)

    @property
    def A(self) -> torch.Tensor:
        return torch.view_as_complex(self._A)

    def _kernel(self, L: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)            
        A  = self.A                              

        dtA   = dt[:, None] * A[None, :]        
        bar_A = torch.exp(dtA)                  
        bar_B = ((bar_A - 1) / A[None, :]) * self.B[None, :] 

        t = torch.arange(L, device=dt.device).float()
        bar_A_pow = torch.exp(dtA[:, :, None] * t[None, None, :]) 
        CB = (self.C * bar_B)[:, :, None]                           
        K  = (CB * bar_A_pow).sum(dim=1).real
        return K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, H)  →  y : (B, L, H)"""
        B, L, H = x.shape
        K = self._kernel(L)                        

        x_f = torch.fft.rfft(x.permute(0, 2, 1), n=2 * L)   
        K_f = torch.fft.rfft(K, n=2 * L)                     
        y_f = x_f * K_f[None]
        y   = torch.fft.irfft(y_f, n=2 * L)[..., :L]        

        y = y + x.permute(0, 2, 1) * self.D[None, :, None]
        return y.permute(0, 2, 1)                             

    def state_matrix_info(self) -> dict:
        A  = self.A.detach()
        dt = torch.exp(self.log_dt).mean().item()
        disc = torch.exp(dt * A)
        return {
            "eig_real":        A.real.numpy(),
            "eig_imag":        A.imag.numpy(),
            "disc_magnitude":  disc.abs().numpy(),
            "spectral_radius": disc.abs().max().item(),
            "stable":          bool((disc.abs() < 1).all()),
        }



class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = int(expand * d_model)
        self.dt_rank  = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1).float()[None].expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D       = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        xz           = self.in_proj(x)                      
        x_in, z      = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_in.permute(0, 2, 1))[..., :L].permute(0, 2, 1)
        x_conv = F.silu(x_conv)

        y = self._ssm(x_conv) * F.silu(z)
        return self.out_proj(y)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective SSM core.  x : (B, L, d_inner)"""
        N = self.d_state
        A = -torch.exp(self.A_log)                          

        proj= self.x_proj(x)                          
        dt_raw, B_mat, C_mat = proj.split([self.dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))               

        return self._scan(x, dt, A, B_mat, C_mat)

    def _scan(self, u, dt, A, B, C) -> torch.Tensor:
        """
        u  : (B, L, d_inner)
        dt : (B, L, d_inner)
        A  : (d_inner, N)
        B  : (B, L, N)
        C  : (B, L, N)
        """
        Bsz, L, Di = u.shape
        N = A.shape[1]

       
        dA = torch.exp(dt[:, :, :, None] * A[None, None])  
        dB = dt[:, :, :, None] * B[:, :, None, :]           

        h   = torch.zeros(Bsz, Di, N, device=u.device)
        ys  = []
        for t in range(L):
            h  = dA[:, t] * h + dB[:, t] * u[:, t, :, None]
            yt = (h * C[:, t, None, :]).sum(-1)             
            ys.append(yt)

        y = torch.stack(ys, dim=1)                          
        return y + u * self.D[None, None, :]

    def state_matrix_info(self) -> dict:
        A_mean = -torch.exp(self.A_log).detach().mean(0)   
        disc   = torch.exp(A_mean)
        return {
            "eig_real":        A_mean.numpy(),
            "eig_imag":        np.zeros(self.d_state),
            "disc_magnitude":  disc.abs().numpy(),
            "spectral_radius": disc.abs().max().item(),
            "stable":          bool((disc.abs() < 1).all()),
        }



def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return F.cosine_similarity(a[None], b[None]).item()


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    ac, bc = a - a.mean(), b - b.mean()
    return (ac * bc).sum().item() / (ac.norm() * bc.norm() + 1e-8)


def power_spectrum(y: torch.Tensor) -> np.ndarray:
    """Mean power spectrum over batch and channel dimensions."""
    signal = y.float().mean(0).mean(-1)           # (L,)
    return torch.fft.rfft(signal).abs().pow(2).detach().numpy()


def impulse_response(model: nn.Module, d_model: int, L: int) -> torch.Tensor:
    x = torch.zeros(1, L, d_model)
    x[0, 0, :] = 1.0
    with torch.no_grad():
        return model(x)


def step_response(model: nn.Module, d_model: int, L: int) -> torch.Tensor:
    x = torch.ones(1, L, d_model)
    with torch.no_grad():
        return model(x)


def benchmark(model: nn.Module, x: torch.Tensor, n: int = 60) -> dict:
    for _ in range(5):
        with torch.no_grad():
            model(x)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        times.append(time.perf_counter() - t0)
    return {
        "mean_ms": np.mean(times) * 1e3,
        "std_ms":  np.std(times)  * 1e3,
        "min_ms":  np.min(times)  * 1e3,
    }


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def compare(
    d_model:      int = 32,
    d_state_s4d:  int = 64,
    d_state_mamba:int = 16,
    seq_len:      int = 128,
    batch_size:   int = 4,
) -> None:
    torch.manual_seed(42)

    s4d   = S4D(d_model=d_model, d_state=d_state_s4d).eval()
    mamba = MambaBlock(d_model=d_model, d_state=d_state_mamba).eval()

    x = torch.randn(batch_size, seq_len, d_model)

    print(f"  Input  : ({batch_size}, {seq_len}, {d_model})")
    print(f"  S4D d_state  = {d_state_s4d}")
    print(f"  Mamba d_state = {d_state_mamba}")

    with torch.no_grad():
        y_s4d   = s4d(x)
        y_mamba = mamba(x)

    section("Output Statistics")
    for name, y in [("S4D", y_s4d), ("Mamba", y_mamba)]:
        print(f"  {name}  shape={tuple(y.shape)}")
        print(f"    mean={y.mean().item():.5f}  std={y.std().item():.5f}"
              f"  min={y.min().item():.5f}  max={y.max().item():.5f}")
        print(f"    L2 norm = {y.norm().item():.5f}")

    section("Output Similarity  (S4D vs Mamba)")
    print(f"  MSE               : {F.mse_loss(y_s4d, y_mamba).item():.6f}")
    print(f"  MAE               : {F.l1_loss (y_s4d, y_mamba).item():.6f}")
    print(f"  Cosine similarity : {cosine_sim(y_s4d, y_mamba):.6f}  "
          f"(1.0 = identical direction)")
    print(f"  Pearson r         : {pearson(y_s4d, y_mamba):.6f}  "
          f"(1.0 = perfect linear relation)")

    
    section("Impulse Response  (unit spike at t=0)")
    ir_s4d   = impulse_response(s4d,   d_model, seq_len)
    ir_mamba = impulse_response(mamba, d_model, seq_len)

    for name, ir in [("S4D", ir_s4d), ("Mamba", ir_mamba)]:
        ch = ir[0, :, 0]
        pk = ch.abs().argmax().item()
        print(f"  {name}:")
        print(f"    peak t={pk}  value={ch[pk].item():.6f}")
        print(f"    final value (t={seq_len-1}) = {ch[-1].item():.6f}")
        print(f"    energy ||h||² = {(ch**2).sum().item():.6f}")

    print(f"\n  Impulse MSE              : {F.mse_loss(ir_s4d, ir_mamba).item():.6f}")
    print(f"  Impulse cosine similarity: {cosine_sim(ir_s4d, ir_mamba):.6f}")

    section("Step Response  (constant input = 1)")
    sr_s4d   = step_response(s4d,   d_model, seq_len)
    sr_mamba = step_response(mamba, d_model, seq_len)

    for name, sr in [("S4D", sr_s4d), ("Mamba", sr_mamba)]:
        ch       = sr[0, :, 0]
        steady   = ch[-1].item()
        peak     = ch.max().item()
        overshoot = (peak - steady) / (abs(steady) + 1e-8) * 100
        print(f"  {name}:")
        print(f"    steady-state = {steady:.6f}")
        print(f"    peak         = {peak:.6f}")
        print(f"    overshoot    = {overshoot:.2f}%")

    section("Spectral (Frequency-Domain) Analysis")
    for name, y in [("S4D", y_s4d), ("Mamba", y_mamba)]:
        ps  = power_spectrum(y)
        print(f"  {name}:")
        print(f"    total power       = {ps.sum():.4f}")
        print(f"    dominant freq bin = {ps.argmax()}")
        print(f"    power in DC (f=0) = {ps[0]:.4f}")
        print(f"    power in top 5 bins: {np.sort(ps)[-5:][::-1].round(3)}")

    section("State-Matrix Eigenvalue & Stability Analysis")
    for name, model in [("S4D", s4d), ("Mamba", mamba)]:
        info = model.state_matrix_info()
        er, ei = info["eig_real"], info["eig_imag"]
        dm     = info["disc_magnitude"]
        print(f"  {name}:")
        print(f"    continuous eigs — real : [{er.min():.4f}, {er.max():.4f}]")
        print(f"    continuous eigs — imag : [{ei.min():.4f}, {ei.max():.4f}]")
        print(f"    discrete  eig magnitudes: [{dm.min():.4f}, {dm.max():.4f}]")
        print(f"    spectral radius : {info['spectral_radius']:.6f}")
        print(f"    BIBO stable     : {info['stable']}")

    section("Parameter Count")
    n_s4d   = count_params(s4d)
    n_mamba = count_params(mamba)
    print(f"  S4D   : {n_s4d:>10,} parameters")
    print(f"  Mamba : {n_mamba:>10,} parameters")
    print(f"  Ratio (Mamba / S4D) : {n_mamba / n_s4d:.2f}×")

    section("Inference-Time Benchmark  (60 runs, CPU)")
    bm_s4d   = benchmark(s4d,   x)
    bm_mamba = benchmark(mamba, x)
    print(f"  S4D   — mean {bm_s4d['mean_ms']:.3f} ms  "
          f"± {bm_s4d['std_ms']:.3f} ms  "
          f"(min {bm_s4d['min_ms']:.3f} ms)")
    print(f"  Mamba — mean {bm_mamba['mean_ms']:.3f} ms  "
          f"± {bm_mamba['std_ms']:.3f} ms  "
          f"(min {bm_mamba['min_ms']:.3f} ms)")
    print(f"  Speed ratio (Mamba / S4D) : {bm_mamba['mean_ms'] / bm_s4d['mean_ms']:.2f}×")

if __name__ == "__main__":
    compare()

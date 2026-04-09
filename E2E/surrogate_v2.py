import math
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Baseline.fda_rof_funcs_v2 import (
    load_parabits_mat,
    qpskmod_with_qammod,
    crmapping,
    giins,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

##基础工具
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return RoundSTE.apply(x)


def round_ste_complex(x: torch.Tensor) -> torch.Tensor:
    xr = round_ste(x.real)
    xi = round_ste(x.imag)
    return torch.complex(xr, xi)


def complex_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y) ** 2)


def complex_awgn(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    x: complex tensor, shape [L]
    snr_db: signal-to-noise ratio in dB
    """
    sig_power = torch.mean(torch.abs(x) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear

    noise_r = torch.randn_like(x.real) * torch.sqrt(noise_power / 2)
    noise_i = torch.randn_like(x.imag) * torch.sqrt(noise_power / 2)
    noise = torch.complex(noise_r, noise_i)
    return x + noise
## torch resample
def design_lowpass_fir(num_taps: int, cutoff: float, device=device):
    """
    cutoff: normalized cutoff in (0, 0.5], relative to sampling rate
    returns real FIR tensor, shape [num_taps]
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps should be odd for symmetric FIR.")

    n = torch.arange(num_taps, dtype=torch.float64, device=device)
    m = n - (num_taps - 1) / 2

    # ideal sinc low-pass
    h = 2 * cutoff * torch.sinc(2 * cutoff * m)

    # Hann window
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * n / (num_taps - 1))
    h = h * w

    # normalize DC gain
    h = h / torch.sum(h)
    return h
def complex_fir_conv1d(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    x: complex [L]
    h: real [K]
    return: complex [L] (same-length convolution)
    """
    x = x.reshape(1, 1, -1)  # [B,C,L]
    h = h.reshape(1, 1, -1)

    pad = (h.shape[-1] - 1) // 2

    yr = F.conv1d(x.real, h, padding=pad)
    yi = F.conv1d(x.imag, h, padding=pad)

    y = torch.complex(yr.squeeze(0).squeeze(0), yi.squeeze(0).squeeze(0))
    return y
def upsample_1d(x: torch.Tensor, up: int) -> torch.Tensor:
    if up == 1:
        return x
    L = x.numel()
    y = torch.zeros(L * up, dtype=x.dtype, device=x.device)
    y[::up] = x
    return y


def downsample_1d(x: torch.Tensor, down: int) -> torch.Tensor:
    if down == 1:
        return x
    return x[::down]
#%%
def resample_fir_complex(
    x: torch.Tensor,
    up: int,
    down: int,
    num_taps: int = 63,
) -> torch.Tensor:
    """
    Approximate resample_poly for complex signals:
    1) upsample
    2) low-pass FIR
    3) downsample
    """
    if up <= 0 or down <= 0:
        raise ValueError("up and down must be positive integers")

    x = x.reshape(-1)

    if up == 1 and down == 1:
        return x

    # upsample
    y = upsample_1d(x, up)

    # anti-imaging / anti-alias cutoff
    cutoff = 0.5 / max(up, down)
    h = design_lowpass_fir(num_taps=num_taps, cutoff=cutoff, device=x.device)

    # gain compensation for zero insertion
    h = h * up

    # filter
    y = complex_fir_conv1d(y, h)

    # downsample
    y = downsample_1d(y, down)

    return y
## DA Surrogate
def c_da_mod_torch(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, order: int):
    """
    Torch version of C_DA_Mod for complex 1D signal.

    Args:
        x: complex tensor, shape [L]
        a: scalar tensor
        b: scalar tensor
        order: int

    Returns:
        tdm_sig: complex tensor, shape [(order+1)*L]
        digital_parts: complex tensor, shape [order, L]
        analog_part: complex tensor, shape [L]
    """
    x = x.reshape(-1)
    L = x.numel()

    digital_parts = []
    e_work = x

    for _ in range(order):
        q = round_ste_complex(e_work * a) / a
        d = q * b
        digital_parts.append(d)
        e_work = (e_work - d / b) * (2.0 * a)

    analog_part = e_work

    if order > 0:
        digital_parts = torch.stack(digital_parts, dim=0)   # [order, L]
        sig = torch.cat([digital_parts, analog_part.unsqueeze(0)], dim=0)  # [order+1, L]
    else:
        digital_parts = torch.zeros((0, L), dtype=x.dtype, device=x.device)
        sig = analog_part.unsqueeze(0)

    # MATLAB-like column-major flatten:
    # original shape [order+1, L]
    # want columns stacked => transpose first, then flatten in row-major
    tdm_sig = sig.transpose(0, 1).reshape(-1)

    return tdm_sig, digital_parts, analog_part
def c_da_demod_torch(rx_sig: torch.Tensor, a: torch.Tensor, b: torch.Tensor, order: int):
    """
    Torch version of C_DA_DeMod for complex 1D signal.

    Args:
        rx_sig: complex tensor, shape [(order+1)*L]
        a: scalar tensor
        b: scalar tensor
        order: int

    Returns:
        x_hat: complex tensor, shape [L]
        digital_sum: complex tensor, shape [L]
        analog_rec: complex tensor, shape [L]
    """
    rx_sig = rx_sig.reshape(-1)
    block = order + 1

    if rx_sig.numel() % block != 0:
        raise ValueError("Length of rx_sig must be divisible by order+1.")

    L = rx_sig.numel() // block

    # inverse of mod packing
    demux = rx_sig.reshape(L, block).transpose(0, 1)   # [order+1, L]

    digital_sum = torch.zeros(L, dtype=rx_sig.dtype, device=rx_sig.device)

    for i in range(order):
        d_i = round_ste_complex(a * demux[i] / b) / a
        d_i = d_i / ((2.0 * a) ** i)
        digital_sum = digital_sum + d_i

    analog_rec = demux[order] / ((2.0 * a) ** order)
    x_hat = digital_sum + analog_rec

    return x_hat, digital_sum, analog_rec
#%%
def get_rt_params(Rt: float):
    """
    Match the original code:
        p = ceil(Rt) - Rt
        frac = Fraction(float(p)).limit_denominator(10**6)
        m, n = frac.numerator, frac.denominator
        N1 = floor(Rt) - 1
        N2 = ceil(Rt) - 1
    """
    p = math.ceil(Rt) - Rt
    frac = Fraction(float(p)).limit_denominator(10**6)
    m, n = frac.numerator, frac.denominator

    N1 = int(math.floor(Rt) - 1)
    N2 = int(math.ceil(Rt) - 1)

    if m <= 0 or n <= 0:
        raise ValueError(f"Invalid m, n from Rt={Rt}: m={m}, n={n}")

    return {
        "p": p,
        "q": Rt - math.floor(Rt),
        "m": m,
        "n": n,
        "N1": N1,
        "N2": N2,
    }


#%%
def split_signal_by_rt(x: torch.Tensor, Rt: float):
    """
    Split x into segment1 and segment2 blockwise according to Rt.

    For each block of length n:
      - first m samples -> seg1
      - remaining (n-m) samples -> seg2

    Only full blocks are kept, matching the original code.
    """
    x = x.reshape(-1)
    cfg = get_rt_params(Rt)
    m, n = cfg["m"], cfg["n"]

    total_blocks = x.numel() // n
    usable_len = total_blocks * n
    x_use = x[:usable_len]

    if total_blocks == 0:
        seg1 = torch.zeros(0, dtype=x.dtype, device=x.device)
        seg2 = torch.zeros(0, dtype=x.dtype, device=x.device)
        return seg1, seg2, cfg, total_blocks

    blocks = x_use.reshape(total_blocks, n)  # [B, n]
    seg1 = blocks[:, :m].reshape(-1)
    seg2 = blocks[:, m:].reshape(-1)

    return seg1, seg2, cfg, total_blocks
#%%
def merge_signal_by_rt(seg1: torch.Tensor, seg2: torch.Tensor, Rt: float):
    """
    Merge segment1 and segment2 back into full signal according to Rt.
    """
    seg1 = seg1.reshape(-1)
    seg2 = seg2.reshape(-1)

    cfg = get_rt_params(Rt)
    m, n = cfg["m"], cfg["n"]

    if m == 0:
        raise ValueError("m must be positive.")

    total_blocks = seg1.numel() // m
    expected_seg2_len = total_blocks * (n - m)

    if seg2.numel() < expected_seg2_len:
        raise ValueError(
            f"seg2 too short: got {seg2.numel()}, expected at least {expected_seg2_len}"
        )

    seg1_use = seg1[: total_blocks * m].reshape(total_blocks, m)
    seg2_use = seg2[: expected_seg2_len].reshape(total_blocks, n - m)

    merged = torch.cat([seg1_use, seg2_use], dim=1).reshape(-1)
    return merged, total_blocks
#%%
def normalize_by_maxabs(x: torch.Tensor, eps: float = 1e-12):
    x = x.reshape(-1)
    if x.numel() == 0:
        return x, torch.tensor(1.0, dtype=torch.float64, device=x.device)

    max_abs = torch.max(torch.abs(x))
    scale = torch.where(max_abs > eps, max_abs, torch.tensor(1.0, dtype=torch.float64, device=x.device))
    x_norm = x / scale
    return x_norm, scale

def continuous_da_mod_torch(
    ofdm_sig: torch.Tensor,
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
    Rt: float,
):
    """
    Torch version of Continous_DA_Mod.
    """
    ofdm_sig,_ = normalize_by_maxabs(ofdm_sig) # normalize input to prevent overflow in DA mod
    seg1, seg2, cfg, total_blocks = split_signal_by_rt(ofdm_sig, Rt)

    ofdm_seg1 = seg1.clone()
    ofdm_seg2 = seg2.clone()

    seg1_norm, scale1 = normalize_by_maxabs(seg1)
    seg2_norm, scale2 = normalize_by_maxabs(seg2)

    N1 = cfg["N1"]
    N2 = cfg["N2"]

    sig_n1, _, _ = c_da_mod_torch(seg1, a1, b1, N1)
    sig_n2, _, _ = c_da_mod_torch(seg2, a2, b2, N2)

    aux = {
        "cfg": cfg,
        "total_blocks": total_blocks,
        "scale1": scale1,
        "scale2": scale2,
        "ofdm_seg1": ofdm_seg1,
        "ofdm_seg2": ofdm_seg2,
        "seg1_norm": seg1,
        "seg2_norm": seg2,
    }

    return sig_n1, sig_n2, aux
#%%
def continuous_da_demod_torch(
    rx_sig_n1: torch.Tensor,
    rx_sig_n2: torch.Tensor,
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
    Rt: float,
    scale1: torch.Tensor = None,
    scale2: torch.Tensor = None,
):
    """
    Torch version of Continous_DA_DeMod with optional scale recovery.
    """
    cfg = get_rt_params(Rt)
    N1 = cfg["N1"]
    N2 = cfg["N2"]

    rx_seg1, _, _ = c_da_demod_torch(rx_sig_n1, a1, b1, N1)
    rx_seg2, _, _ = c_da_demod_torch(rx_sig_n2, a2, b2, N2)

    if scale1 is not None:
        rx_seg1 = rx_seg1
    if scale2 is not None:
        rx_seg2 = rx_seg2

    rx_ofdm, total_blocks = merge_signal_by_rt(rx_seg1, rx_seg2, Rt)

    aux = {
        "cfg": cfg,
        "total_blocks": total_blocks,
        "rx_seg1": rx_seg1,
        "rx_seg2": rx_seg2,
    }

    return rx_ofdm, rx_seg1, rx_seg2, aux
class ContinuousDABlockTorch(nn.Module):
    def __init__(self, Rt: float, init_a1: float, init_b1: float, init_a2: float, init_b2: float, eps: float = 1e-4):
        super().__init__()
        self.Rt = Rt
        self.eps = eps
        self.cfg = get_rt_params(Rt)

        def inv_softplus(y):
            y = max(y, eps)
            return math.log(math.exp(y) - 1.0)

        self.theta_a1 = nn.Parameter(torch.tensor(inv_softplus(init_a1), dtype=torch.float64))
        self.theta_b1 = nn.Parameter(torch.tensor(inv_softplus(init_b1), dtype=torch.float64))
        self.theta_a2 = nn.Parameter(torch.tensor(inv_softplus(init_a2), dtype=torch.float64))
        self.theta_b2 = nn.Parameter(torch.tensor(inv_softplus(init_b2), dtype=torch.float64))

    @property
    def a1(self):
        return F.softplus(self.theta_a1) + self.eps

    @property
    def b1(self):
        return F.softplus(self.theta_b1) + self.eps

    @property
    def a2(self):
        return F.softplus(self.theta_a2) + self.eps

    @property
    def b2(self):
        return F.softplus(self.theta_b2) + self.eps

    def mod(self, ofdm_sig: torch.Tensor):
        return continuous_da_mod_torch(ofdm_sig, self.a1, self.b1, self.a2, self.b2, self.Rt)

    def demod(self, rx_sig_n1: torch.Tensor, rx_sig_n2: torch.Tensor, scale1=None, scale2=None):
        return continuous_da_demod_torch(
            rx_sig_n1, rx_sig_n2,
            self.a1, self.b1, self.a2, self.b2,
            self.Rt,
            scale1=scale1,
            scale2=scale2,
        )

    def forward(self, ofdm_sig: torch.Tensor):
        sig_n1, sig_n2, mod_aux = self.mod(ofdm_sig)
        x_hat, rx_seg1, rx_seg2, demod_aux = self.demod(
            sig_n1, sig_n2,
            scale1=mod_aux["scale1"],
            scale2=mod_aux["scale2"],
        )
        return x_hat, sig_n1, sig_n2, mod_aux, demod_aux
## FDM+config
def build_fdm_config_from_rt(
    Rt: float,
    OFDM_bandwidth: float = 5e9,
    G: float = 2e9,
    sps: int = 8,
):
    """
    Build FDM/resampling config following the original NumPy code.
    """
    p = math.ceil(Rt) - Rt
    q = Rt - math.floor(Rt)
    N1 = int(math.floor(Rt) - 1)
    N2 = int(math.ceil(Rt) - 1)

    B0 = OFDM_bandwidth
    B1 = B0 * p * (N1 + 1)
    B2 = B0 * q * (N2 + 1)

    if B1 <= B2:
        Sys_Symbolrate = B2
        up_tx_1 = int(round((B2 / B1) * sps))
        down_tx_1 = sps
        up_tx_2 = 1
        down_tx_2 = 1

        up_rx_1 = sps
        down_rx_1 = int(round((B2 / B1) * sps))
        up_rx_2 = 1
        down_rx_2 = 1
    else:
        Sys_Symbolrate = B1
        up_tx_1 = 1
        down_tx_1 = 1
        up_tx_2 = int(round((B1 / B2) * sps))
        down_tx_2 = sps

        up_rx_1 = 1
        down_rx_1 = 1
        up_rx_2 = sps
        down_rx_2 = int(round((B1 / B2) * sps))

    Sys_bandwidth = B1 + B2

    f_inner = B1 / 2.0 + G
    f_outer = f_inner + B1 / 2.0 + B2 / 2.0 + G

    cfg = {
        "Rt": Rt,
        "p": p,
        "q": q,
        "N1": N1,
        "N2": N2,
        "B0": B0,
        "B1": B1,
        "B2": B2,
        "Sys_Symbolrate": Sys_Symbolrate,
        "Sys_bandwidth": Sys_bandwidth,
        "G": G,
        "f_inner_hz": f_inner,
        "f_outer_hz": f_outer,
        "sps": sps,
        "up_tx_1": up_tx_1,
        "down_tx_1": down_tx_1,
        "up_tx_2": up_tx_2,
        "down_tx_2": down_tx_2,
        "up_rx_1": up_rx_1,
        "down_rx_1": down_rx_1,
        "up_rx_2": up_rx_2,
        "down_rx_2": down_rx_2,
    }
    return cfg
#%%
def add_discrete_freqs_to_fdm_config(cfg):
    Fs = cfg["sps"] * cfg["Sys_Symbolrate"]
    cfg = dict(cfg)
    cfg["Fs_hz"] = Fs
    cfg["f_inner_norm"] = cfg["f_inner_hz"] / Fs
    cfg["f_outer_norm"] = cfg["f_outer_hz"] / Fs
    return cfg
#%%
def complex_exp_carrier(length: int, freq_norm: float, device=device):
    """
    freq_norm: normalized discrete-time frequency in cycles/sample
               e.g. 0.1 means exp(j*2*pi*0.1*n)
    """
    n = torch.arange(length, dtype=torch.float64, device=device)
    phase = 2.0 * math.pi * freq_norm * n
    return torch.exp(1j * phase).to(torch.complex128)


def upconvert(x: torch.Tensor, freq_norm: float):
    x = x.reshape(-1)
    c = complex_exp_carrier(x.numel(), freq_norm, device=x.device)
    return x * c


def downconvert(x: torch.Tensor, freq_norm: float):
    x = x.reshape(-1)
    c = complex_exp_carrier(x.numel(), freq_norm, device=x.device)
    return x * torch.conj(c)
def pad_or_crop_to_length(x: torch.Tensor, target_len: int):
    x = x.reshape(-1)
    L = x.numel()

    if L == target_len:
        return x
    elif L < target_len:
        y = torch.zeros(target_len, dtype=x.dtype, device=x.device)
        y[:L] = x
        return y
    else:
        return x[:target_len]


def match_lengths(x1: torch.Tensor, x2: torch.Tensor, mode: str = "pad_to_max"):
    """
    mode:
      - pad_to_max: zero-pad shorter one
      - crop_to_min: crop longer one
    """
    L1 = x1.numel()
    L2 = x2.numel()

    if mode == "pad_to_max":
        L = max(L1, L2)
    elif mode == "crop_to_min":
        L = min(L1, L2)
    else:
        raise ValueError("mode must be 'pad_to_max' or 'crop_to_min'")

    return pad_or_crop_to_length(x1, L), pad_or_crop_to_length(x2, L), L
#%%
def fdm_combine_from_cfg(sig_n1_rs: torch.Tensor,
                         sig_n2_rs: torch.Tensor,
                         k: torch.Tensor,
                         cfg: dict,
                         length_mode: str = "pad_to_max"):
    """
    Follow original code:
      inner <- branch 1
      outer <- branch 2
    """
    sig_inner, sig_outer, L = match_lengths(sig_n1_rs, sig_n2_rs, mode=length_mode)

    tx_inner = upconvert(sig_inner, cfg["f_inner_norm"]) * k
    tx_outer = upconvert(sig_outer, cfg["f_outer_norm"])

    tx_fdm = tx_inner + tx_outer

    aux = {
        "L_fdm": L,
        "tx_inner": tx_inner,
        "tx_outer": tx_outer,
    }
    return tx_fdm, aux


def fdm_separate_from_cfg(rx_fdm: torch.Tensor,
                          k: torch.Tensor,
                          cfg: dict):
    """
    Follow original RX code:
      RX_Signal_original_N1 = (1/k) * RxSignal * exp(-j2π f_inner t)
      RX_Signal_original_N2 = RxSignal * exp(-j2π f_outer t)
    """

    rx_fdm,_ = normalize_by_maxabs(rx_fdm) # ensure max abs is exactly 1.0 for stable downconversion
    rx_inner = downconvert(rx_fdm, cfg["f_inner_norm"]) / k
    rx_outer = downconvert(rx_fdm, cfg["f_outer_norm"])
    return rx_inner, rx_outer
#%%
def branch_resample_tx_from_cfg(sig_n1: torch.Tensor,
                                sig_n2: torch.Tensor,
                                cfg: dict,
                                num_taps: int = 63):
    tx1 = resample_fir_complex(
        sig_n1,
        up=cfg["up_tx_1"],
        down=cfg["down_tx_1"],
        num_taps=num_taps
    )
    tx2 = resample_fir_complex(
        sig_n2,
        up=cfg["up_tx_2"],
        down=cfg["down_tx_2"],
        num_taps=num_taps
    )
    return tx1, tx2


def branch_resample_rx_from_cfg(rx1: torch.Tensor,
                                rx2: torch.Tensor,
                                cfg: dict,
                                num_taps: int = 63):
    rec1 = resample_fir_complex(
        rx1,
        up=cfg["up_rx_1"],
        down=cfg["down_rx_1"],
        num_taps=num_taps
    )
    rec2 = resample_fir_complex(
        rx2,
        up=cfg["up_rx_2"],
        down=cfg["down_rx_2"],
        num_taps=num_taps
    )
    return rec1, rec2
#positive parameter
class PositiveScalar(nn.Module):
    def __init__(self, init_value: float, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

        def inv_softplus(y):
            y = max(y, eps)
            return math.log(math.exp(y) - 1.0)

        self.theta = nn.Parameter(torch.tensor(inv_softplus(init_value), dtype=torch.float64))

    @property
    def value(self):
        return F.softplus(self.theta) + self.eps
# RRC
def rrc_impulse_torch(beta: float, sps: int, span: int, device=device):
    """
    Torch version of the original rrc_impulse(beta, sps, span).
    Returns a real FIR tensor of shape [span*sps + 1].
    """
    N = span * sps
    t = torch.arange(-N / 2, N / 2 + 1, dtype=torch.float64, device=device) / sps
    h = torch.zeros_like(t, dtype=torch.float64)

    for i, ti in enumerate(t):
        ti_val = ti.item()

        if abs(ti_val) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / math.pi

        elif beta != 0 and abs(abs(ti_val) - 1.0 / (4.0 * beta)) < 1e-12:
            h[i] = (beta / math.sqrt(2.0)) * (
                (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * beta))
                + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * beta))
            )

        else:
            numerator = (
                math.sin(math.pi * ti_val * (1.0 - beta))
                + 4.0 * beta * ti_val * math.cos(math.pi * ti_val * (1.0 + beta))
            )
            denominator = math.pi * ti_val * (1.0 - (4.0 * beta * ti_val) ** 2)
            h[i] = numerator / denominator

    # normalize energy, matching the original code
    h = h / torch.sqrt(torch.sum(h ** 2))
    return h
#%%
def complex_fir_lfilter(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Causal FIR filtering, intended to mimic scipy.signal.lfilter(h, [1.0], x)
    for FIR h.

    x: complex [L]
    h: real [K]
    return: complex [L]
    """
    x = x.reshape(-1)
    h = h.reshape(-1)

    # conv1d does correlation, so flip h to emulate convolution/lfilter FIR
    h_flip = torch.flip(h, dims=[0]).reshape(1, 1, -1)

    xr = x.real.reshape(1, 1, -1)
    xi = x.imag.reshape(1, 1, -1)

    K = h.numel()
    # causal FIR: left-pad with K-1 zeros
    xr_pad = F.pad(xr, (K - 1, 0))
    xi_pad = F.pad(xi, (K - 1, 0))

    yr = F.conv1d(xr_pad, h_flip).reshape(-1)
    yi = F.conv1d(xi_pad, h_flip).reshape(-1)

    return torch.complex(yr, yi)
#%%
def apply_rrc_tx_torch(x: torch.Tensor, beta: float, sps: int, span: int):
    """
    Torch version of original apply_rrc_tx.
    Output length = len(x) * sps
    """
    x = x.reshape(-1)

    xpad = torch.cat([
        x,
        torch.zeros(span // 2, dtype=x.dtype, device=x.device)
    ], dim=0)

    h = rrc_impulse_torch(beta=beta, sps=sps, span=span, device=x.device)

    # manual upsample
    upsampled = torch.zeros(xpad.numel() * sps, dtype=x.dtype, device=x.device)
    upsampled[::sps] = xpad

    y = complex_fir_lfilter(upsampled, h)

    delay = (span * sps) // 2
    y = y[delay:]

    y = y[: x.numel() * sps]
    return y
# RRC
def apply_rrc_rx_torch(x: torch.Tensor, beta: float, sps: int, span: int):
    """
    Torch version of original apply_rrc_rx.
    Output length = len(x)
    """
    x = x.reshape(-1)

    xpad = torch.cat([
        x,
        torch.zeros((span * sps) // 2, dtype=x.dtype, device=x.device)
    ], dim=0)

    h = rrc_impulse_torch(beta=beta, sps=sps, span=span, device=x.device)

    y = complex_fir_lfilter(xpad, h)

    delay = (span * sps) // 2
    y = y[delay:]

    y = y[: x.numel()]
    return y
## OFDM generation

OFDM_CFG = {
    "para": 256,
    "fftlen": 256,
    "nd": 40,
    "ml": 6,       # 64QAM
    "gilen": 32,
}
OFDM_CFG["fftlen2"] = OFDM_CFG["fftlen"] + OFDM_CFG["gilen"]
def build_ofdm_from_bits(bits, cfg=OFDM_CFG):
    """
    bits: 1D numpy array, shape [para * nd * ml]
    return: dict with OFDM-domain signal and intermediate results
    """
    para   = cfg["para"]
    fftlen = cfg["fftlen"]
    nd     = cfg["nd"]
    ml     = cfg["ml"]
    gilen  = cfg["gilen"]
    fftlen2 = cfg["fftlen2"]

    total_bits = para * nd * ml
    bits = np.asarray(bits).astype(int).reshape(-1)

    if bits.size != total_bits:
        raise ValueError(f"bits length mismatch: got {bits.size}, expected {total_bits}")

    # Match original code: column-major reshape
    paradata = bits.reshape(para, nd * ml, order="F")

    # 64QAM modulation via original helper
    QAM_I, QAM_Q = qpskmod_with_qammod(paradata, para, nd, ml)

    # Original normalization for 64QAM average power = 42
    QAM_I_norm = QAM_I / np.sqrt(42.0)
    QAM_Q_norm = QAM_Q / np.sqrt(42.0)

    Signal_qam = QAM_I_norm + 1j * QAM_Q_norm
    Signal_qam_serial = Signal_qam.reshape(para * nd, order="F")

    # OFDM mapping + IFFT
    xI, xQ = crmapping(QAM_I_norm, QAM_Q_norm, fftlen, nd)
    x = xI + 1j * xQ
    y = np.fft.ifft(x, n=fftlen, axis=0)

    # Add cyclic prefix
    ich_cp, qch_cp = giins(np.real(y), np.imag(y), fftlen, gilen, nd)
    Signal_gi = ich_cp + 1j * qch_cp

    # Serialize OFDM frames
    Signal_serial = Signal_gi.reshape(fftlen2 * nd, order="F")
    Signal_OFDM = Signal_serial.reshape(-1)

    return {
        "bits": bits,
        "paradata": paradata,
        "QAM_I_norm": QAM_I_norm,
        "QAM_Q_norm": QAM_Q_norm,
        "Signal_qam_serial": Signal_qam_serial,
        "Signal_OFDM": Signal_OFDM,
        "para": para,
        "nd": nd,
        "ml": ml,
        "fftlen": fftlen,
        "gilen": gilen,
        "fftlen2": fftlen2,
    }



## SNR calculation

def real_mmse_gain(tx: torch.Tensor, rx: torch.Tensor, eps=1e-12):
    """
    Find real scalar h minimizing || h * tx - rx ||^2

    h* = Re{sum(conj(tx)*rx)} / sum(|tx|^2)
    """
    denom = torch.sum(torch.abs(tx) ** 2).real + eps
    numer = torch.real(torch.sum(torch.conj(tx) * rx))
    h = numer / denom
    return h


def snr_db_paper_style(tx: torch.Tensor, rx: torch.Tensor, eps=1e-12):
    """
    Paper-style recovered SNR:
        1) find real scalar h minimizing ||h*tx - rx||^2
        2) signal power  Ps = mean(|h*tx|^2)
        3) error power   Pn = mean(|rx - h*tx|^2)
        4) SNR_dB = 10 log10(Ps / Pn)

    No separate normalization of tx and rx here.
    """
    h = real_mmse_gain(tx, rx, eps=eps)

    ref = h * tx
    err = rx - ref

    Ps = torch.mean(torch.abs(ref) ** 2)
    Pn = torch.mean(torch.abs(err) ** 2)

    snr_linear = Ps / (Pn + eps)
    snr_db = 10.0 * torch.log10(snr_linear + eps)

    aux = {
        "h": h.detach().item(),
        "Ps": Ps.detach().item(),
        "Pn": Pn.detach().item(),
        "snr_linear": snr_linear.detach().item(),
    }
    return snr_db, aux


def compute_surrogate_snr_loss(model, x, seg_weight=0.2):
    """
    x: complex tensor [L]
    objective:
        maximize total recovered SNR
        optionally add segment-level auxiliary SNR terms
    """
    x_hat, rx_seg1, rx_seg2, aux = model(x)

    usable_len = aux["mod_aux"]["total_blocks"] * aux["mod_aux"]["cfg"]["n"]
    x_use = x[:usable_len]

    ref_seg1 = aux["mod_aux"]["ofdm_seg1"]
    ref_seg2 = aux["mod_aux"]["ofdm_seg2"]

    snr_total_db, aux_total = snr_db_paper_style(x_use, x_hat)
    snr_seg1_db, aux_seg1 = snr_db_paper_style(ref_seg1, rx_seg1)
    snr_seg2_db, aux_seg2 = snr_db_paper_style(ref_seg2, rx_seg2)

    # main objective: maximize total SNR
    # optional: use seg_weight to stabilize training
    loss = -(snr_total_db )

    stats = {
        "loss": loss.detach().item(),
        "snr_total_db": snr_total_db.detach().item(),
        "snr_seg1_db": snr_seg1_db.detach().item(),
        "snr_seg2_db": snr_seg2_db.detach().item(),
        "usable_len": usable_len,
        "total_h": aux_total["h"],
        "total_Ps": aux_total["Ps"],
        "total_Pn": aux_total["Pn"],
        "seg1_h": aux_seg1["h"],
        "seg1_Ps": aux_seg1["Ps"],
        "seg1_Pn": aux_seg1["Pn"],
        "seg2_h": aux_seg2["h"],
        "seg2_Ps": aux_seg2["Ps"],
        "seg2_Pn": aux_seg2["Pn"],
    }
    return loss, stats
## waveform loss
def compute_surrogate_waveform_loss(model, x, seg_weight=0.2, use_nmse=False, eps=1e-12):
    """
    x: complex tensor [L]

    waveform-level training objective:
        main: make reconstructed OFDM waveform x_hat close to target waveform x_use
        aux : make reconstructed segment waveforms close to reference segments

    SNR is still computed and returned for monitoring, but is NOT the training loss.

    Args:
        model: surrogate model
        x: complex input waveform [L]
        seg_weight: weight for segment-level auxiliary losses
        use_nmse: if True, use normalized MSE; otherwise use plain MSE
        eps: numerical stability
    """
    x_hat, rx_seg1, rx_seg2, aux = model(x)

    usable_len = aux["mod_aux"]["total_blocks"] * aux["mod_aux"]["cfg"]["n"]
    x_use = x[:usable_len]

    ref_seg1 = aux["mod_aux"]["ofdm_seg1"]
    ref_seg2 = aux["mod_aux"]["ofdm_seg2"]

    # -------- waveform loss helpers --------
    def complex_nmse(ref: torch.Tensor, rec: torch.Tensor, eps=1e-12):
        num = torch.mean(torch.abs(rec - ref) ** 2)
        den = torch.mean(torch.abs(ref) ** 2) + eps
        return num / den

    if use_nmse:
        loss_total = complex_nmse(x_use, x_hat, eps=eps)
        loss_seg1 = complex_nmse(ref_seg1, rx_seg1, eps=eps)
        loss_seg2 = complex_nmse(ref_seg2, rx_seg2, eps=eps)
    else:
        loss_total = complex_mse(x_use, x_hat)
        loss_seg1 = complex_mse(ref_seg1, rx_seg1)
        loss_seg2 = complex_mse(ref_seg2, rx_seg2)

    loss = loss_total

    # -------- SNR for monitoring only --------
    snr_total_db, aux_total = snr_db_paper_style(x_use, x_hat, eps=eps)
    snr_seg1_db, aux_seg1 = snr_db_paper_style(ref_seg1, rx_seg1, eps=eps)
    snr_seg2_db, aux_seg2 = snr_db_paper_style(ref_seg2, rx_seg2, eps=eps)

    stats = {
        "loss": loss.detach().item(),
        "loss_total": loss_total.detach().item(),
        "loss_seg1": loss_seg1.detach().item(),
        "loss_seg2": loss_seg2.detach().item(),

        "snr_total_db": snr_total_db.detach().item(),
        "snr_seg1_db": snr_seg1_db.detach().item(),
        "snr_seg2_db": snr_seg2_db.detach().item(),

        "usable_len": usable_len,

        "total_h": aux_total["h"],
        "total_Ps": aux_total["Ps"],
        "total_Pn": aux_total["Pn"],

        "seg1_h": aux_seg1["h"],
        "seg1_Ps": aux_seg1["Ps"],
        "seg1_Pn": aux_seg1["Pn"],

        "seg2_h": aux_seg2["h"],
        "seg2_Ps": aux_seg2["Ps"],
        "seg2_Pn": aux_seg2["Pn"],
    }
    return loss, stats
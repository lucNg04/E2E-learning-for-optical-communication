# fda_rof_funcs_v1.py
import numpy as np
from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple, Dict, Any

from scipy.signal import upfirdn, resample_poly, firwin, lfilter
import matplotlib.pyplot as plt


# =========================================================
# Utilities
# =========================================================
def rat_approx(x: float, tol: float = 1e-6, max_den: int = 10_000_000) -> Tuple[int, int]:
    """Approximate x by m/n (MATLAB rat-like)."""
    if x <= 0:
        return 0, 1
    den_caps = [1024, 8192, 65536, 1_000_000, max_den]
    best = None
    for cap in den_caps:
        frac = Fraction(x).limit_denominator(cap)
        err = abs(x - frac.numerator / frac.denominator)
        if best is None or err < best[0]:
            best = (err, frac.numerator, frac.denominator)
        if err <= tol:
            return frac.numerator, frac.denominator
    _, m, n = best
    return m, n


def matlab_round_like(x: np.ndarray) -> np.ndarray:
    """MATLAB-like round: half away from zero; complex -> round real/imag separately."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        r = np.sign(np.real(x)) * np.floor(np.abs(np.real(x)) + 0.5)
        im = np.sign(np.imag(x)) * np.floor(np.abs(np.imag(x)) + 0.5)
        return r + 1j * im
    else:
        return np.sign(x) * np.floor(np.abs(x) + 0.5)


# =========================================================
# QAM (bit input), used by your qpskmod_with_qammod wrapper
# =========================================================
def _gray_code(n: np.ndarray) -> np.ndarray:
    return n ^ (n >> 1)

def _bits_to_integers(bits: np.ndarray, k: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int64)
    if bits.ndim == 1:
        bits = bits.reshape((-1, k))
    weights = (1 << np.arange(k - 1, -1, -1, dtype=np.int64))
    return (bits * weights).sum(axis=1)

def qammod_bits_gray_square(bits_2d: np.ndarray, M: int) -> np.ndarray:
    """
    Square Gray-coded M-QAM mapper (classic ±1,±3,... on I/Q).
    For 64QAM average energy is 42, matching your later normalization by sqrt(42).
    bits_2d: (Nsym, log2(M))
    """
    k = int(np.log2(M))
    m_side = int(np.sqrt(M))
    if m_side * m_side != M or 2**k != M or k % 2 != 0:
        raise ValueError("Only square M-QAM with even log2(M) is supported.")

    sym_int = _bits_to_integers(bits_2d, k)
    k2 = k // 2
    i_int = sym_int >> k2
    q_int = sym_int & ((1 << k2) - 1)

    i_gray = _gray_code(i_int)
    q_gray = _gray_code(q_int)

    levels = np.arange(-(m_side - 1), m_side, 2, dtype=np.float64)
    I = levels[i_gray]
    Q = levels[q_gray]
    return I + 1j * Q


def qpskmod_with_qammod(paradata: np.ndarray, para: int, nd: int, ml: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python equivalent of your MATLAB qpskmod_with_qammod (actually M-QAM).
    paradata: (para, nd*ml) bits
    output iout/qout: (para, nd)
    """
    paradata = np.asarray(paradata, dtype=np.int64)
    if paradata.shape != (para, nd * ml):
        raise ValueError(f"Expected paradata shape {(para, nd*ml)}, got {paradata.shape}")

    M = 2 ** ml
    iout = np.zeros((para, nd), dtype=np.float64)
    qout = np.zeros((para, nd), dtype=np.float64)

    for i in range(para):
        row_bits = paradata[i, :].reshape((nd, ml))
        modulated = qammod_bits_gray_square(row_bits, M)  # (nd,)
        iout[i, :] = np.real(modulated)
        qout[i, :] = np.imag(modulated)

    return iout, qout


# =========================================================
# OFDM helpers (as per your MATLAB)
# =========================================================
def crmapping(idata: np.ndarray, qdata: np.ndarray, fftlen: int, nd: int) -> Tuple[np.ndarray, np.ndarray]:
    """Your current MATLAB version is identity mapping."""
    iout = np.zeros((fftlen, nd), dtype=np.float64)
    qout = np.zeros((fftlen, nd), dtype=np.float64)
    iout[:fftlen, :] = idata[:fftlen, :]
    qout[:fftlen, :] = qdata[:fftlen, :]
    return iout, qout


def giins(idata: np.ndarray, qdata: np.ndarray, fftlen: int, gilen: int, nd: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cyclic prefix insertion; returns (fftlen+gilen, nd)."""
    idata1 = np.asarray(idata).reshape((fftlen, nd))
    qdata1 = np.asarray(qdata).reshape((fftlen, nd))
    iout = np.vstack([idata1[fftlen - gilen:fftlen, :], idata1])
    qout = np.vstack([qdata1[fftlen - gilen:fftlen, :], qdata1])
    return iout, qout
def girem(ich: np.ndarray, qch: np.ndarray, fftlen2: int, gilen: int, nd: int):
    """
    Remove cyclic prefix. Input ich/qch represent (fftlen2, nd) or flattened length fftlen2*nd in column-major.
    Returns ich5/qch5 shaped (fftlen, nd).
    """
    ich = np.asarray(ich).reshape(-1)
    qch = np.asarray(qch).reshape(-1)
    if ich.size != fftlen2 * nd or qch.size != fftlen2 * nd:
        # allow (fftlen2, nd) matrix input
        ich = np.asarray(ich).reshape((fftlen2, nd), order="F").reshape(-1, order="F")
        qch = np.asarray(qch).reshape((fftlen2, nd), order="F").reshape(-1, order="F")

    x = (ich + 1j * qch).reshape((fftlen2, nd), order="F")
    x_nocp = x[gilen:, :]  # remove first gilen rows
    return np.real(x_nocp), np.imag(x_nocp)


def crdemapping(ich: np.ndarray, qch: np.ndarray, fftlen: int, nd: int):
    """
    RX demapping matching your current TX crmapping() (identity).
    """
    ich = np.asarray(ich).reshape((fftlen, nd))
    qch = np.asarray(qch).reshape((fftlen, nd))
    return ich.copy(), qch.copy()


def gray_to_binary(g: np.ndarray) -> np.ndarray:
    """
    Vectorized Gray->binary conversion for non-negative integers.
    """
    g = g.astype(np.int64, copy=False)
    b = np.zeros_like(g)
    shift = g.copy()
    while np.any(shift):
        b ^= shift
        shift >>= 1
    return b


def qamdemod_bits_gray_square(symbols: np.ndarray, M: int) -> np.ndarray:
    """
    Hard-decision demapper for square Gray-coded M-QAM.
    Returns bits with shape (Nsym, log2(M)).
    """
    symbols = np.asarray(symbols).reshape(-1)
    k = int(np.log2(M))
    m_side = int(np.sqrt(M))
    if m_side * m_side != M or k % 2 != 0:
        raise ValueError("Only square M-QAM with even log2(M) is supported.")

    k2 = k // 2
    levels = np.arange(-(m_side - 1), m_side, 2, dtype=np.float64)

    I = np.real(symbols)
    Q = np.imag(symbols)

    # nearest level index
    i_idx = np.argmin(np.abs(I[:, None] - levels[None, :]), axis=1)
    q_idx = np.argmin(np.abs(Q[:, None] - levels[None, :]), axis=1)

    # indices are Gray-coded in your mod (you used gray_code -> levels[gray])
    i_bin = gray_to_binary(i_idx)
    q_bin = gray_to_binary(q_idx)

    sym_int = (i_bin << k2) | q_bin

    # integer -> bits (MSB first)
    bits = ((sym_int[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.int64)
    return bits


def OFDM_DeMod(PNC_RC_RxSignal: np.ndarray, fftlen2: int, gilen: int, nd: int,
              fftlen: int, para: int, ml: int):
    """
    MATLAB OFDM_DeMod equivalent (for your current identity mapping case).
    Returns:
      PNC_RC_RxSignal_sym (para*nd, 1) complex vector (QAM symbols),
      demodata_sequence (para*nd*ml, 1) bit vector
    """
    x = np.asarray(PNC_RC_RxSignal).reshape(-1)
    # Expect input as (fftlen2, nd) matrix OR flattened column-major
    if x.size == fftlen2 * nd:
        x_mat = x.reshape((fftlen2, nd), order="F")
    else:
        # maybe already matrix
        x_mat = np.asarray(PNC_RC_RxSignal)
        if x_mat.shape != (fftlen2, nd):
            raise ValueError(f"Expected {(fftlen2, nd)} or length {fftlen2*nd}, got {x_mat.shape} / {x.size}")

    ich4 = np.real(x_mat).reshape(-1, order="F")
    qch4 = np.imag(x_mat).reshape(-1, order="F")

    ich5, qch5 = girem(ich4, qch4, fftlen2, gilen, nd)
    rx = ich5 + 1j * qch5  # (fftlen, nd)

    ry = np.fft.fft(rx, n=fftlen, axis=0)
    ich6 = np.real(ry)
    qch6 = np.imag(ry)

    ich7, qch7 = crdemapping(ich6, qch6, fftlen, nd)

    kmod = np.sqrt(42.0)  # for 64QAM normalization you used in TX
    ich8 = ich7 / kmod
    qch8 = qch7 / kmod

    PNC_RC_RxSignal_sym_mat = ich8 + 1j * qch8  # (fftlen, nd)
    # You use para=256==fftlen. If later para<fftlen, slice here:
    PNC_RC_RxSignal_sym_mat = PNC_RC_RxSignal_sym_mat[:para, :]

    # MATLAB: reshape(PNC_RC_RxSignal, para*nd, 1) column-major
    PNC_RC_RxSignal_sym = PNC_RC_RxSignal_sym_mat.reshape((para * nd, 1), order="F")

    # demap to bits
    bits = qamdemod_bits_gray_square(PNC_RC_RxSignal_sym_mat.reshape(-1, order="F"), M=2**ml)
    demodata_sequence = bits.reshape((para * nd * ml, 1), order="F")  # bits already row-major per symbol

    return PNC_RC_RxSignal_sym, demodata_sequence


# =========================================================
# FDA / IDA core
# =========================================================
@dataclass
class DAParam:
    a: float
    b: float


def C_DA_Mod(E: np.ndarray, a: float, b: float, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact translation of your MATLAB C_DA_Mod.
    Output Signal_out is a column vector of shape ((order+1)*len, 1).
    """
    E = np.asarray(E).reshape(-1)
    L = E.size

    Digital_part = np.zeros((L, order), dtype=E.dtype)

    for i in range(order):
        Digital_part[:, i] = matlab_round_like(E * a) * (1.0 / a) * b
        E = (E - Digital_part[:, i] / b) * (2.0 * a)
        if i == order - 1:
            Analog_part = E

    # Build sig exactly: rows = segments, each row length L
    sig = np.vstack([Digital_part.T, Analog_part.reshape(1, -1)])  # (order+1, L)

    # MATLAB: reshape(sig,1,[]) on a row-stacked matrix => row-wise flatten
    TDM_Sig = sig.reshape(-1, order='F')  # row-wise
    Signal_out = TDM_Sig.reshape(-1, 1)   # column vector
    return Signal_out, Digital_part
def C_DA_DeMod(Signal_in: np.ndarray, a: float, b: float, order: int) -> np.ndarray:
    """
    Inverse of C_DA_Mod (MATLAB-equivalent).
    Signal_in: column-like vector of length (order+1)*L (TDM interleaved by rows in MOD).
    Returns: recovered E0 of length L (1D).
    """
    x = np.asarray(Signal_in).reshape(-1)
    if order < 0:
        raise ValueError("order must be >= 0")

    if order == 0:
        # In MOD: sig = [Analog_part] only
        return x.copy()

    seg = order + 1
    if x.size % seg != 0:
        raise ValueError(f"Input length {x.size} not divisible by (order+1)={seg}")

    L = x.size // seg

    # MOD did: sig = vstack([Digital_part.T, Analog_part]) then row-wise flatten (order='C')
    sig = x.reshape((seg, L), order="F")  # row-major, matches your MOD flatten

    digital = sig[:order, :]              # shape (order, L)
    analog  = sig[order, :]               # shape (L,)

    # Backward recursion: E = D_i/b + E/(2a)
    E = analog.astype(np.complex128, copy=False)
    for i in range(order - 1, -1, -1):
        E = (digital[i, :] / b) + (E / (2.0 * a))

    return E
import numpy as np

def getN0_MMSE(signal_ref, signal_rx):
    """
    计算 MMSE 噪声功率 (N0)。
    """
    error_signal = signal_rx - signal_ref
    N0 = np.mean(np.abs(error_signal)**2) / np.mean(np.abs(signal_ref)**2)
    return N0, error_signal

def Metric_Calculation(RxSignal, TxSignal, title):
    """
    计算信号的 SNR。
    """

    RxSignal = RxSignal.flatten()
    Signal_matched_len2 = TxSignal[:len(RxSignal)].flatten()

    N0_OFDM, _ = getN0_MMSE(Signal_matched_len2, RxSignal)
    SNR_OFDM = 10 * np.log10((1 - N0_OFDM) / N0_OFDM)

    print(f"SNR of {title} symbols is {SNR_OFDM:.2f} dB")
    return SNR_OFDM

def Continous_DA_Mod(OFDM_Sig: np.ndarray, para_N1: DAParam, para_N2: DAParam, Rt: float):
    """
    Translation of your MATLAB Continous_DA_Mod:
    - p = ceil(Rt)-Rt, [m,n] = rat(p,1e-6)
    - sample-wise partition blocks of length n: first m -> sig1, rest -> sig2
    - normalize each segment by max(abs(.))
    - C_DA_Mod on each segment
    """
    OFDM_Sig = np.asarray(OFDM_Sig).reshape(-1)

    p = np.ceil(Rt) - Rt
    m, n = rat_approx(float(p), tol=1e-6)

    N1 = int(np.floor(Rt) - 1)
    N2 = int(np.ceil(Rt) - 1)

    sig1 = []
    sig2 = []
    total_blocks = int(len(OFDM_Sig) // n)

    for i in range(total_blocks):
        block = OFDM_Sig[i * n:(i + 1) * n]
        if m > 0:
            sig1.append(block[:m])
        if m < n:
            sig2.append(block[m:])

    sig1 = np.concatenate(sig1) if len(sig1) else np.array([], dtype=OFDM_Sig.dtype)
    sig2 = np.concatenate(sig2) if len(sig2) else np.array([], dtype=OFDM_Sig.dtype)

    OFDM_Segment1 = sig1.copy()
    OFDM_Segment2 = sig2.copy()

    if sig1.size > 0:
        sig1 = sig1 / (np.max(np.abs(sig1)) + 1e-30)
    if sig2.size > 0:
        sig2 = sig2 / (np.max(np.abs(sig2)) + 1e-30)

    Sig_N1, _ = C_DA_Mod(sig1, para_N1.a, para_N1.b, N1) if sig1.size > 0 else (np.zeros((0, 1), dtype=np.complex128), None)
    Sig_N2, _ = C_DA_Mod(sig2, para_N2.a, para_N2.b, N2) if sig2.size > 0 else (np.zeros((0, 1), dtype=np.complex128), None)

    return Sig_N1, Sig_N2, OFDM_Segment1, OFDM_Segment2
def Continous_DA_DeMod(
    RX_Signal_N1: np.ndarray,
    RX_Signal_N2: np.ndarray,
    para_N1: DAParam,
    para_N2: DAParam,
    Rt: float,
):
    """
    MATLAB Continous_DA_DeMod equivalent.

    Returns:
      Rx_OFDM_Sig (1D),
      RX_OFDM_Sig_Segment1 (1D),
      RX_OFDM_Sig_Segment2 (1D)
    """
    print("Rx_Signal_N1 len =", len(RX_Signal_N1))
    print("Rx_Signal_N2 len =", len(RX_Signal_N2))
    p = float(np.ceil(Rt) - Rt)
    m, n = rat_approx(p, tol=1e-6)
    N1 = int(np.floor(Rt) - 1)
    N2 = int(np.ceil(Rt) - 1)

    # N1 < N2 assumed (same as your MATLAB comment)
    seg1 = C_DA_DeMod(RX_Signal_N1, para_N1.a, para_N1.b, N1)
    seg2 = C_DA_DeMod(RX_Signal_N2, para_N2.a, para_N2.b, N2)

    seg1 = np.asarray(seg1).reshape(-1)
    seg2 = np.asarray(seg2).reshape(-1)
    # p = float(np.ceil(Rt) - Rt)
    # m, n = rat_approx(p, tol=1e-6)
    # N1 = int(np.floor(Rt) - 1)
    # N2 = int(np.ceil(Rt) - 1)
    #
    # segN1 = N1 + 1
    # segN2 = N2 + 1
    #
    # RX_Signal_N1 = np.asarray(RX_Signal_N1).reshape(-1)
    # RX_Signal_N2 = np.asarray(RX_Signal_N2).reshape(-1)
    #
    # RX_Signal_N1 = RX_Signal_N1[: (len(RX_Signal_N1) // segN1) * segN1]
    # RX_Signal_N2 = RX_Signal_N2[: (len(RX_Signal_N2) // segN2) * segN2]
    #
    # seg1 = C_DA_DeMod(RX_Signal_N1, para_N1.a, para_N1.b, N1)
    # seg2 = C_DA_DeMod(RX_Signal_N2, para_N2.a, para_N2.b, N2)
    if m == 0:
        # p=0 (integer Rt) -> no segment1 in each block
        total_blocks = seg2.size // n
        Rx = seg2[: total_blocks * n].copy()
        return Rx, seg1, seg2

    total_blocks_1 = seg1.size // m
    total_blocks_2 = seg2.size // (n - m)
    total_blocks = min(total_blocks_1, total_blocks_2)
    #total_blocks = seg1.size // m
    Rx = np.zeros(n * total_blocks, dtype=np.complex128)

    idx = 0
    for i in range(total_blocks):
        s1 = seg1[i * m : (i + 1) * m]
        s2 = seg2[i * (n - m) : (i + 1) * (n - m)]
        Rx[idx : idx + n] = np.concatenate([s1, s2])
        idx += n
    print("len seg1 =", len(seg1))
    print("len seg2 =", len(seg2))
    print("m, n =", m, n)
    print("Rx_OFDM_Sig len =", len(Rx))

    return Rx, seg1, seg2


# =========================================================
# RRC (SRRC) filter + apply (TX)
# =========================================================
def rrc_impulse(beta: float, sps: int, span: int) -> np.ndarray:
    """Root raised cosine impulse response; unit-energy normalization."""
    n_taps = span * sps + 1
    t = (np.arange(n_taps) - n_taps // 2) / sps
    h = np.zeros_like(t, dtype=np.float64)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + (4 * beta / np.pi)
        elif beta != 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / (den + 1e-30)

    h = h / np.sqrt(np.sum(h**2) + 1e-30)
    return h


# def apply_rrc_tx(x: np.ndarray, beta: float, sps: int, span: int) -> np.ndarray:
#     """
#     Match MATLAB comm.RaisedCosineTransmitFilter('Square root'):
#     - upsample by sps and filter
#     - remove group delay: (span*sps)/2 samples
#     """
#     x = np.asarray(x).reshape(-1)
#     h = rrc_impulse(beta, sps, span)
#     y = upfirdn(h, x, up=sps)
#     delay = (span * sps) // 2
#     y = y[delay:]
#     return y
def apply_rrc_tx(x: np.ndarray, beta: float, sps: int, span: int) -> np.ndarray:
    x = np.asarray(x).reshape(-1)

    # 对应 MATLAB: [Sig; zeros(order/2,1)]
    x_pad = np.concatenate([x, np.zeros(span // 2, dtype=x.dtype)])

    h = rrc_impulse(beta, sps, span)
    y = upfirdn(h, x_pad, up=sps)

    delay = (span * sps) // 2
    y = y[delay:]

    return y

# =========================================================
# Plot spectrum (your plotRxSpectrum_2)
# =========================================================
def plotRxSpectrum_2(signal: np.ndarray, Fs: float, SymbolRate: float, sps: int, Title: str):
    signal = np.asarray(signal).reshape(-1)
    Fs_in = sps * SymbolRate

    up, down = rat_approx(Fs / Fs_in, tol=1e-12)
    sig = resample_poly(signal, up=up, down=down)

    N = len(sig)
    RX = np.fft.fftshift(np.fft.fft(sig, N))
    f = np.linspace(-Fs/2, Fs/2, N, endpoint=False)
    PSD = 10*np.log10((np.abs(RX)**2)/(N + 1e-30) + 1e-30)

    plt.figure()
    plt.plot(f/1e9, PSD, linewidth=1.2)
    plt.grid(True)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Power Spectrum of {Title} (Fs = {Fs/1e9:.0f} GHz)")
    plt.xlim([-Fs/2/1e9, Fs/2/1e9])
    plt.show()


# =========================================================
# setSimulationParams (your function)
# =========================================================
def setSimulationParams(sampRate: float, nSamples: int) -> Dict[str, Any]:
    tWindow = nSamples / sampRate
    dt = 1.0 / sampRate
    df = sampRate / nSamples
    t = np.arange(nSamples) * dt
    f = (np.arange(-nSamples/2, nSamples/2) * (sampRate / nSamples))
    return dict(sampRate=sampRate, nSamples=nSamples, tWindow=tWindow, df=df, dt=dt, t=t, f=f)


# =========================================================
# High-level TX pipeline: FDA Tx after OFDM is prepared
# =========================================================

def tx_fda_pipeline(
    Signal_OFDM: np.ndarray,
    Rt: float,
    OFDM_bandwidth: float,
    para_N1: DAParam,
    para_N2: DAParam,
    roll: float,
    span: int,
    sps: int,
    G: float,
    k_power: float,
    AwgSampleRate: float,
    export_awg: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    N1 locked as INNER sub-band:
      Continous_DA_Mod -> RRC -> bandwidth-align resample -> FDM shift -> sum
      -> normalize I/Q -> (optional) resample to AWG rate

    Notes:
      - For fractional Rt, N1 < N2 always holds, so "N1 inner" is consistent.
      - For integer Rt, p=0 and the N1 branch may be empty; a small safeguard is included.
    """
    # FDA split + C_DA_Mod
    Sig_N1, Sig_N2, seg1, seg2 = Continous_DA_Mod(Signal_OFDM, para_N1, para_N2, Rt)
    Sig_N1 = Sig_N1.reshape(-1)
    Sig_N2 = Sig_N2.reshape(-1)

    # RRC shaping (TX)
    s1 = apply_rrc_tx(Sig_N1, beta=roll, sps=sps, span=span)
    s2 = apply_rrc_tx(Sig_N2, beta=roll, sps=sps, span=span)

    # Bandwidths / orders
    p = np.ceil(Rt) - Rt
    q = Rt - np.floor(Rt)
    N1 = int(np.floor(Rt) - 1)
    N2 = int(np .ceil(Rt) - 1)

    B0 = OFDM_bandwidth
    B1 = B0 * p * (N1 + 1)
    B2 = B0 * q * (N2 + 1)

    # Align sample rates to Fs_base = sps * max(B1, B2)
    # (do NOT decide inner/outer here; we lock N1 as inner afterwards)
    if B1 <= B2:
        Sys_Symbolrate = B2
        if B1 > 0:
            up, down = rat_approx(B2 / (B1 + 1e-30), tol=1e-12)
            s1a = resample_poly(s1, up=up, down=down)  # N1 up to N2 rate
        else:
            s1a = s1
        s2a = s2
    else:
        Sys_Symbolrate = B1
        if B2 > 0:
            up, down = rat_approx(B1 / (B2 + 1e-30), tol=1e-12)
            s2a = resample_poly(s2, up=up, down=down)  # N2 up to N1 rate
        else:
            s2a = s2
        s1a = s1

    Fs_base = sps * Sys_Symbolrate
    # Lock N1 as inner (your requirement)
    Sig_inner, Sig_outer = s1a, s2a
    B_inner, B_outer = B1, B2
    which_inner = "N1"

    # Safeguard for integer Rt / p=0 where N1 branch can be empty
    if Sig_inner.size == 0:
        Sig_inner, Sig_outer = s2a, np.zeros_like(s2a)
        B_inner, B_outer = B2, 0.0
        which_inner = "N2_only"
    L_inner = int(len(Sig_inner))
    L_outer = int(len(Sig_outer))

    # Zero pad
    L = max(len(Sig_inner), len(Sig_outer))
    Sig_inner = np.pad(Sig_inner, (0, L - len(Sig_inner)))
    Sig_outer = np.pad(Sig_outer, (0, L - len(Sig_outer)))

    # Frequency plan (same structure as your MATLAB)
    f_inner = B_inner / 2 + G
    f_outer = f_inner + B_inner / 2 + B_outer / 2 + G

    PARAM = setSimulationParams(Fs_base, L)
    t = PARAM["t"]

    Sig_inner_fdm = (k_power * Sig_inner) * np.exp(1j * 2 * np.pi * f_inner * t)
    Sig_outer_fdm = Sig_outer * np.exp(1j * 2 * np.pi * f_outer * t)

    Stx_Sig = Sig_inner_fdm + Sig_outer_fdm  # base-rate complex waveform

    # Normalize I/Q jointly
    I = np.real(Stx_Sig)
    Q = np.imag(Stx_Sig)
    max_value = max(np.max(np.abs(I)), np.max(np.abs(Q)), 1e-30)
    I = I / max_value
    Q = Q / max_value

    if export_awg:
        up_awg, down_awg = rat_approx(AwgSampleRate / Fs_base, tol=1e-12)
        I_out = resample_poly(I, up=up_awg, down=down_awg)
        Q_out = resample_poly(Q, up=up_awg, down=down_awg)
    else:
        I_out, Q_out = I, Q

    meta = dict(
        Rt=Rt, p=float(p), q=float(q), N1=N1, N2=N2,
        B0=B0, B1=float(B1), B2=float(B2),
        Sys_Symbolrate=float(Sys_Symbolrate), Fs_base=float(Fs_base),
        B_inner=float(B_inner), B_outer=float(B_outer), which_inner=which_inner,
        f_inner=float(f_inner), f_outer=float(f_outer),
        L_base=int(L),
        seg1_len=int(len(seg1)), seg2_len=int(len(seg2)),
        L_sig_inner=L_inner,
        L_sig_outer=L_outer,

    )
    return I_out, Q_out, Stx_Sig, meta



##Channel

from typing import Optional, Dict, Any, Tuple

def awgn_measured(x: np.ndarray, snr_db: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    MATLAB: awgn(x, SNR, 'measured') equivalent for complex baseband.
    SNR is defined w.r.t. measured average signal power.
    """
    x = np.asarray(x).reshape(-1)
    if rng is None:
        rng = np.random.default_rng()

    sig_power = np.mean(np.abs(x) ** 2)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_lin

    n = (rng.normal(scale=np.sqrt(noise_power / 2), size=x.shape) +
         1j * rng.normal(scale=np.sqrt(noise_power / 2), size=x.shape))
    return x + n


def channel_pipeline(
    Stx_Sig: np.ndarray,
    SNR: Optional[float] = None,
    enable_awgn: bool = False,
    rng: Optional[np.random.Generator] = None,
    # placeholders for future expansion
    enable_tx_phase_noise: bool = False,
    enable_cd: bool = False,
    enable_lo_phase_noise: bool = False,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Channel block (minimal version aligned with your MATLAB script).

    Default behavior (like your current MATLAB final line):
        RxSignal = Stx_Sig

    Optional:
        enable_awgn=True -> RxSignal = awgn(Stx_Sig, SNR, 'measured')
    """
    x = np.asarray(Stx_Sig).reshape(-1)

    if enable_tx_phase_noise:
        raise NotImplementedError("TX phase noise (Laser_Noise) not implemented yet.")

    if enable_cd:
        raise NotImplementedError("Chromatic dispersion (CDInsertion) not implemented yet.")

    if enable_awgn:
        if SNR is None:
            raise ValueError("SNR must be provided when enable_awgn=True.")
        x = awgn_measured(x, snr_db=float(SNR), rng=rng)

    if enable_lo_phase_noise:
        raise NotImplementedError("LO phase noise (Laser_Noise) not implemented yet.")

    meta_ch = dict(
        enable_awgn=bool(enable_awgn),
        SNR=float(SNR) if SNR is not None else None,
        enable_tx_phase_noise=bool(enable_tx_phase_noise),
        enable_cd=bool(enable_cd),
        enable_lo_phase_noise=bool(enable_lo_phase_noise),
        n_samples=int(x.size),
    )
    return x, meta_ch

## receiver
def apply_rrc_rx(x: np.ndarray, beta: float, sps: int, span: int) -> np.ndarray:
    """
    RX matched filter to match MATLAB comm.RaisedCosineReceiveFilter('Square root', DecimationFactor=1).
    We mimic your MATLAB usage:
      y = rctFilt([x; zeros(span*sps/2)])
      y = y(span*sps/2 : end)
    """
    x = np.asarray(x).reshape(-1)
    h = rrc_impulse(beta, sps, span)  # same SRRC as TX
    gd = (span * sps) // 2
    y = lfilter(h, [1.0], np.concatenate([x, np.zeros(gd)]))
    y = y[gd:]  # remove group delay
    return y


def rx_fda_pipeline_base(
    RxSignal: np.ndarray,
    meta: dict,
    Rt: float,
    OFDM_bandwidth: float,
    para_N1: DAParam,
    para_N2: DAParam,
    roll: float,
    span: int,
    sps: int,
    G: float,
    k_power: float,
    fftlen: int,
    gilen: int,
    nd: int,
    para: int,
    ml: int,
    L_sig_inner: int,
    L_sig_outer: int,
):
    """
    Base-rate RX pipeline (NO extra resample).
    Assumes RxSignal is at Fs_base = meta['Fs_base'].
    Mirrors your MATLAB receiver up to OFDM_DeMod.

    Returns:
      out dict with intermediates + demod bits.
    """
    out = {}
    y = np.asarray(RxSignal).reshape(-1)

    # --- derive subband plan (use same formula as MATLAB) ---
    p = np.ceil(Rt) - Rt
    q = Rt - np.floor(Rt)
    N1 = int(np.floor(Rt) - 1)
    N2 = int(np.ceil(Rt) - 1)
    B0 = OFDM_bandwidth
    B1 = B0 * p * (N1 + 1)
    B2 = B0 * q * (N2 + 1)

    Sys_Symbolrate = max(B1, B2)
    Fs_base = float(meta.get("Fs_base", sps * Sys_Symbolrate))  # prefer meta

    f_inner = B1 / 2 + G
    f_outer = f_inner + B1 / 2 + B2 / 2 + G

    out.update(dict(B1=float(B1), B2=float(B2), Fs_base=Fs_base,
                    f_inner=float(f_inner), f_outer=float(f_outer)))

    # --- time axis at base rate ---
    nSamples = y.size
    t = np.arange(nSamples) / Fs_base

    # --- inverse frequency shifting (N1 locked as inner) ---
    # TX: inner had k_power factor; RX divides by k_power on inner branch
    RX_Signal_original_N1 = (1.0 / k_power) * y * np.exp(-1j * 2 * np.pi * f_inner * t)
    RX_Signal_original_N2 = y * np.exp(-1j * 2 * np.pi * f_outer * t)
    print("B2/B1 =", B2/B1 if B1 > 0 else float('inf'))
    # --- trim to original lengths ---
    RX_Signal_original_N1 = RX_Signal_original_N1[:L_sig_inner]
    RX_Signal_original_N2 = RX_Signal_original_N2[:L_sig_outer]

    # --- inverse of TX bandwidth-align resample ---
    if B1 <= B2:
        # TX had upsampled N1 to B2-rate; RX should bring N1 back
        up, down = rat_approx(B1 / (B2 + 1e-30), tol=1e-12)
        RX_Signal_original_N1 = resample_poly(RX_Signal_original_N1, up=up, down=down)
    else:
        # TX had upsampled N2 to B1-rate; RX should bring N2 back
        up, down = rat_approx(B2 / (B1 + 1e-30), tol=1e-12)
        RX_Signal_original_N2 = resample_poly(RX_Signal_original_N2, up=up, down=down)

    out["rx_n1_shifted"] = RX_Signal_original_N1
    out["rx_n2_shifted"] = RX_Signal_original_N2

    # --- RX SRRC matched filter ---
    RX_Signal_N1 = apply_rrc_rx(RX_Signal_original_N1, beta=roll, sps=sps, span=span)
    RX_Signal_N2 = apply_rrc_rx(RX_Signal_original_N2, beta=roll, sps=sps, span=span)

    out["rx_n1_rrc"] = RX_Signal_N1
    out["rx_n2_rrc"] = RX_Signal_N2

    # --- downsample to 1 sample/symbol ---
    RX_Signal_N1_ds = RX_Signal_N1[::sps]
    RX_Signal_N2_ds = RX_Signal_N2[::sps]

    out["rx_n1_sym"] = RX_Signal_N1_ds
    out["rx_n2_sym"] = RX_Signal_N2_ds
    print("len rx_n1_shifted =", len(RX_Signal_original_N1))
    print("len rx_n2_shifted =", len(RX_Signal_original_N2))
    print("len rx_n1_rrc     =", len(RX_Signal_N1))
    print("len rx_n2_rrc     =", len(RX_Signal_N2))
    print("len rx_n1_sym     =", len(RX_Signal_N1_ds))
    print("len rx_n2_sym     =", len(RX_Signal_N2_ds))
    # --- CDA demod (uses functions we added earlier) ---
    Rx_OFDM_Sig, seg1, seg2 = Continous_DA_DeMod(RX_Signal_N1_ds, RX_Signal_N2_ds, para_N1, para_N2, Rt)
    out["Rx_OFDM_Sig"] = Rx_OFDM_Sig

    out["seg1"] = seg1
    out["seg2"] = seg2

    # --- OFDM demod ---
    fftlen2 = fftlen + gilen
    # MATLAB receiver did: reshape(Rx_OFDM_Sig, fftlen2, nd)
    Rx_OFDM_Sig = np.asarray(Rx_OFDM_Sig).reshape(-1)
    need = fftlen2 * nd
    if Rx_OFDM_Sig.size < need:
        raise ValueError(f"Rx_OFDM_Sig too short: {Rx_OFDM_Sig.size} < {need}")
    Rx_OFDM_Sig_reshape = Rx_OFDM_Sig[:need].reshape((fftlen2, nd), order="F")

    PNC_RC_RxSignal_sym, demodata_sequence = OFDM_DeMod(
        Rx_OFDM_Sig_reshape, fftlen2, gilen, nd, fftlen, para, ml
    )

    out["PNC_RC_RxSignal_sym"] = PNC_RC_RxSignal_sym
    out["demodata_sequence"] = demodata_sequence
    return out

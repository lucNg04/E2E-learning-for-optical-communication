import numpy as np
from fractions import Fraction
from scipy.io import loadmat
from scipy.signal import upfirdn, resample_poly
from scipy.optimize import minimize
from scipy.signal import lfilter


def load_parabits_mat(mat_path="parabits.mat"):
    mat = loadmat(mat_path)
    if "parabits" not in mat:
        raise KeyError(f"'parabits' not found in {mat_path}")
    parabits = np.asarray(mat["parabits"]).astype(int).reshape(-1, order="F")
    return parabits


def setSimulationParams(*args):

    if len(args) == 1:
        PARAM = args[0]
        sampRate = PARAM["sampRate"]
        nSamples = PARAM["nSamples"]
    elif len(args) == 2:
        sampRate = args[0]
        nSamples = args[1]
    else:
        raise ValueError("setSimulationParams expects 1 or 2 arguments.")

    tWindow = nSamples / sampRate
    dt = 1 / sampRate
    df = sampRate / nSamples
    t = np.arange(nSamples) * dt
    f = np.arange(-nSamples / 2, nSamples / 2) * (sampRate / nSamples)

    PARAM = {
        "sampRate": sampRate,
        "nSamples": nSamples,
        "tWindow": tWindow,
        "df": df,
        "dt": dt,
        "t": t,
        "f": f,
    }
    return PARAM


def awgn_measured(x, snr_db):
    """
    MATLAB-like awgn(x, snr_db, 'measured')
    """
    x = np.asarray(x).astype(np.complex128).reshape(-1)
    sig_power = np.mean(np.abs(x) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear

    noise = (
        np.sqrt(noise_power / 2.0)
        * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
    )
    return x + noise


# =========================================================
# RRC filter helpers
# =========================================================

def rrc_impulse(beta, sps, span):
    """
    Root Raised Cosine impulse response
    span: filter span in symbols
    sps : samples per symbol
    """
    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps
    h = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta != 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-12:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            numerator = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            )
            denominator = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = numerator / denominator

    # normalize energy
    h = h / np.sqrt(np.sum(h**2))
    return h


def apply_rrc_tx(x, beta, sps, span):
    """
    MATLAB-like:
        rctFilt = comm.RaisedCosineTransmitFilter(...)
        y = rctFilt([x; zeros(span/2,1)]);
        y = y((span*sps)/2 + 1 : end);

    这里显式控制输出长度为 len(x) * sps
    """

    x = np.asarray(x).reshape(-1)
    xpad = np.concatenate([x, np.zeros(span // 2, dtype=x.dtype)])

    h = rrc_impulse(beta, sps, span)

    # 手动上采样
    upsampled = np.zeros(len(xpad) * sps, dtype=np.complex128)
    upsampled[::sps] = xpad

    # 用 lfilter，而不是 full convolution
    y = lfilter(h, [1.0], upsampled)

    delay = (span * sps) // 2
    y = y[delay:]

    # MATLAB 目标长度应为 len(x)*sps
    y = y[: len(x) * sps]
    return y


def apply_rrc_rx(x, beta, sps, span):
    """
    MATLAB-like:
        rctFilt = comm.RaisedCosineReceiveFilter(..., DecimationFactor=1)
        y = rctFilt([x; zeros(order*sps/2,1)]);
        y = y((order*sps)/2 + 1 : end);

    这里显式控制输出长度为 len(x)
    """
    x = np.asarray(x).reshape(-1)
    xpad = np.concatenate([x, np.zeros((span * sps) // 2, dtype=x.dtype)])

    h = rrc_impulse(beta, sps, span)

    # 用 lfilter，而不是 full convolution
    y = lfilter(h, [1.0], xpad)

    delay = (span * sps) // 2
    y = y[delay:]

    # MATLAB 目标长度应与原输入 x 一致
    y = y[: len(x)]
    return y

# =========================================================
# QAM helpers
# =========================================================



def bits_to_integers_msb_first(bits, ml):
    bits = np.asarray(bits).astype(int)
    weights = 2 ** np.arange(ml - 1, -1, -1)
    return bits @ weights


def integers_to_bits_msb_first(symbols, ml):
    symbols = np.asarray(symbols).astype(int).reshape(-1)
    bits = ((symbols[:, None] >> np.arange(ml - 1, -1, -1)) & 1).astype(int)
    return bits


def gray_to_binary(gray):
    gray = np.asarray(gray).astype(int).reshape(-1)
    binary = gray.copy()
    shift = 1
    while True:
        shifted = binary >> shift
        if np.all(shifted == 0):
            break
        binary ^= shifted
        shift <<= 1
    return binary


def binary_to_gray(binary):
    binary = np.asarray(binary).astype(int).reshape(-1)
    return binary ^ (binary >> 1)


def qam_symbol_from_bits_gray(bits_row):
    """
    bits_row: shape (ml,)
    MATLAB-like square QAM Gray mapping:
    - split bits into I half and Q half
    - each half is Gray-coded PAM
    - Q axis goes from top to bottom
    """
    bits_row = np.asarray(bits_row).astype(int).reshape(-1)
    ml = len(bits_row)

    if ml % 2 != 0:
        raise ValueError("Only even ml (square QAM) is supported.")

    k_axis = ml // 2
    m_side = 2 ** k_axis

    levels = np.arange(-(m_side - 1), m_side, 2)   # e.g. [-3,-1,1,3]
    q_levels = levels[::-1]                         # top to bottom: [3,1,-1,-3]

    i_bits_gray = bits_row[:k_axis]
    q_bits_gray = bits_row[k_axis:]

    i_gray_int = bits_to_integers_msb_first(i_bits_gray[None, :], k_axis)[0]
    q_gray_int = bits_to_integers_msb_first(q_bits_gray[None, :], k_axis)[0]

    i_bin_int = gray_to_binary(np.array([i_gray_int]))[0]
    q_bin_int = gray_to_binary(np.array([q_gray_int]))[0]

    I = levels[i_bin_int]
    Q = q_levels[q_bin_int]

    return I + 1j * Q


def qam_constellation_square(M):
    """
    MATLAB-like default Gray-coded square QAM constellation.
    Indexed by natural integer symbol index written in MSB-first bits.
    """
    ml = int(np.log2(M))
    if 2 ** ml != M:
        raise ValueError("M must be a power of 2.")
    if ml % 2 != 0:
        raise ValueError("This function only supports square QAM (even log2(M)).")

    const = np.zeros(M, dtype=complex)
    all_bits = integers_to_bits_msb_first(np.arange(M), ml)  # natural binary labels

    for idx in range(M):
        const[idx] = qam_symbol_from_bits_gray(all_bits[idx])

    return const


def qamdemod_nearest(symbols, M):
    const = qam_constellation_square(M)
    symbols = np.asarray(symbols).reshape(-1)
    dists = np.abs(symbols[:, None] - const[None, :]) ** 2
    idx = np.argmin(dists, axis=1)
    return idx


def qpskmod_with_qammod(paradata, para, nd, ml):
    """
    MATLAB equivalent:
        qammod(current_row.', 2^ml, 'InputType', 'bit')
    """
    paradata = np.asarray(paradata).astype(int)
    if paradata.shape != (para, nd * ml):
        raise ValueError(f"paradata shape must be ({para}, {nd * ml}), got {paradata.shape}")

    iout = np.zeros((para, nd), dtype=float)
    qout = np.zeros((para, nd), dtype=float)

    for i in range(para):
        current_row = paradata[i, :]
        bits_reshaped = current_row.reshape(nd, ml)

        modulated = np.array([qam_symbol_from_bits_gray(b) for b in bits_reshaped])

        iout[i, :] = np.real(modulated)
        qout[i, :] = np.imag(modulated)

    return iout, qout


def qpskdemod_with_qamdemod(idata, qdata, para, nd, ml):
    """
    MATLAB equivalent:
        qamdemod(y, M, 'OutputType', 'bit')
    by nearest-neighbor to MATLAB-like Gray-coded constellation
    """
    demodata = np.zeros((para, ml * nd), dtype=int)
    complex_data = np.asarray(idata) + 1j * np.asarray(qdata)
    M = 2 ** ml

    for i in range(para):
        current_row = complex_data[i, :]
        demodulated_idx = qamdemod_nearest(current_row, M)
        demodulated_bits = integers_to_bits_msb_first(demodulated_idx, ml)
        demodata[i, :] = demodulated_bits.reshape(-1)

    return demodata


def crmapping(idata, qdata, fftlen, nd):
    iout = np.zeros((fftlen, nd), dtype=np.asarray(idata).dtype)
    qout = np.zeros((fftlen, nd), dtype=np.asarray(qdata).dtype)

    iout[:fftlen, :] = np.asarray(idata)[:fftlen, :]
    qout[:fftlen, :] = np.asarray(qdata)[:fftlen, :]

    return iout, qout


def crdemapping(idata, qdata, fftlen, nd):
    iout = np.zeros((fftlen, nd), dtype=np.asarray(idata).dtype)
    qout = np.zeros((fftlen, nd), dtype=np.asarray(qdata).dtype)

    iout[:fftlen, :] = np.asarray(idata)[:fftlen, :]
    qout[:fftlen, :] = np.asarray(qdata)[:fftlen, :]

    return iout, qout


def giins(idata, qdata, fftlen, gilen, nd):
    idata1 = np.asarray(idata).reshape(fftlen, nd, order="F")
    qdata1 = np.asarray(qdata).reshape(fftlen, nd, order="F")

    iout = np.vstack([idata1[fftlen - gilen:fftlen, :], idata1])
    qout = np.vstack([qdata1[fftlen - gilen:fftlen, :], qdata1])

    return iout, qout


def girem(idata, qdata, fftlen2, gilen, nd):
    idata2 = np.asarray(idata).reshape(fftlen2, nd, order="F")
    qdata2 = np.asarray(qdata).reshape(fftlen2, nd, order="F")

    iout = idata2[gilen:fftlen2, :]
    qout = qdata2[gilen:fftlen2, :]

    return iout, qout


def C_DA_Mod(E, a, b, order):
    E = np.asarray(E).reshape(-1).astype(np.complex128)
    length = len(E)

    Digital_part = np.zeros((length, order), dtype=np.complex128)
    E_work = E.copy()

    for i in range(order):
        Digital_part[:, i] = np.round(E_work * a) * (1 / a) * b
        E_work = (E_work - Digital_part[:, i] / b) * (2 * a)

        if i == order - 1:
            Analog_part = E_work.copy()

    sig = np.zeros((order + 1, length), dtype=np.complex128)
    for i in range(order):
        sig[i, :] = Digital_part[:, i]

    sig[order, :] = Analog_part
    TDM_Sig = sig.reshape(-1, order="F")
    Signal_out = TDM_Sig.reshape(-1, 1)

    return Signal_out, Digital_part


def C_DA_DeMod(Rx_Sig, a, b, order):
    E = np.asarray(Rx_Sig).reshape(-1).astype(np.complex128)
    Demux = E.reshape(order + 1, -1, order="F")

    Digital_part = np.zeros(Demux.shape[1], dtype=np.complex128)

    for i in range(order):
        factor1 = (2 * a) ** i
        D = (1 / a) * np.round(a * Demux[i, :] * (1 / b)) / factor1
        Digital_part = Digital_part + D

    factor2 = (2 * a) ** order
    Analog_part = Demux[order, :] / factor2

    Sig = Digital_part + Analog_part
    Signal_out = Sig.reshape(-1, 1)

    return Signal_out


def Continous_DA_Mod(OFDM_Sig, para_N1, para_N2, Rt):
    p = np.ceil(Rt) - Rt
    frac = Fraction(float(p)).limit_denominator(10**6)
    m, n = frac.numerator, frac.denominator

    N1 = int(np.floor(Rt) - 1)
    N2 = int(np.ceil(Rt) - 1)

    OFDM_Sig = np.asarray(OFDM_Sig).reshape(-1)

    sig1 = []
    sig2 = []

    total_blocks = len(OFDM_Sig) // n
    for i in range(total_blocks):
        block = OFDM_Sig[i * n:(i + 1) * n]
        sig1.append(block[:m])
        sig2.append(block[m:])

    sig1 = np.concatenate(sig1) if len(sig1) > 0 else np.array([], dtype=complex)
    sig2 = np.concatenate(sig2) if len(sig2) > 0 else np.array([], dtype=complex)

    OFDM_Segment1 = sig1.copy()
    OFDM_Segment2 = sig2.copy()

    if len(sig1) > 0 and np.max(np.abs(sig1)) > 0:
        sig1 = sig1 / np.max(np.abs(sig1))

    if len(sig2) > 0 and np.max(np.abs(sig2)) > 0:
        sig2 = sig2 / np.max(np.abs(sig2))

    Sig_N1, _ = C_DA_Mod(sig1, para_N1["a"], para_N1["b"], N1)
    Sig_N2, _ = C_DA_Mod(sig2, para_N2["a"], para_N2["b"], N2)

    return Sig_N1.reshape(-1), Sig_N2.reshape(-1), OFDM_Segment1, OFDM_Segment2


def Continous_DA_DeMod(RX_Signal_N1, RX_Signal_N2, para_N1, para_N2, Rt):
    p = np.ceil(Rt) - Rt
    frac = Fraction(float(p)).limit_denominator(10**6)
    m, n = frac.numerator, frac.denominator

    N1 = int(np.floor(Rt) - 1)
    N2 = int(np.ceil(Rt) - 1)

    RX_OFDM_Sig_Segment1 = C_DA_DeMod(RX_Signal_N1, para_N1["a"], para_N1["b"], N1).reshape(-1)
    RX_OFDM_Sig_Segment2 = C_DA_DeMod(RX_Signal_N2, para_N2["a"], para_N2["b"], N2).reshape(-1)

    total_blocks = len(RX_OFDM_Sig_Segment1) // m
    Rx_OFDM_Sig = np.zeros(n * total_blocks, dtype=complex)

    idx = 0
    for i in range(total_blocks):
        s1 = RX_OFDM_Sig_Segment1[i * m:(i + 1) * m]
        s2 = RX_OFDM_Sig_Segment2[i * (n - m):(i + 1) * (n - m)]
        Rx_OFDM_Sig[idx:idx + n] = np.concatenate([s1, s2])
        idx += n

    return Rx_OFDM_Sig, RX_OFDM_Sig_Segment1, RX_OFDM_Sig_Segment2


def OFDM_DeMod(PNC_RC_RxSignal, fftlen2, gilen, nd, fftlen, para, ml):
    ich4 = np.real(PNC_RC_RxSignal)
    qch4 = np.imag(PNC_RC_RxSignal)

    ich5, qch5 = girem(ich4, qch4, fftlen2, gilen, nd)

    rx = ich5 + 1j * qch5
    ry = np.fft.fft(rx, axis=0)
    ich6 = np.real(ry)
    qch6 = np.imag(ry)

    ich7, qch7 = crdemapping(ich6, qch6, fftlen, nd)

    kmod = np.sqrt(42.0)
    ich8 = ich7 / kmod
    qch8 = qch7 / kmod

    PNC_RC_RxSignal = ich8 + 1j * qch8
    PNC_RC_RxSignal = PNC_RC_RxSignal.reshape(para * nd, 1, order="F")

    demodata = qpskdemod_with_qamdemod(ich8, qch8, para, nd, ml)
    demodata_sequence = demodata.reshape(para * nd * ml, 1, order="F")

    return PNC_RC_RxSignal, demodata_sequence


def getN0_MMSE(Stx, Srx):
    Stx = np.asarray(Stx, dtype=np.complex128)
    Srx = np.asarray(Srx, dtype=np.complex128)

    if Stx.ndim == 1:
        Stx = Stx.reshape(1, -1)
    if Srx.ndim == 1:
        Srx = Srx.reshape(1, -1)

    nPol = Stx.shape[0]
    N0 = np.full(nPol, np.nan, dtype=np.float64)
    c = np.full(nPol, np.nan, dtype=np.float64)

    for n in range(nPol):
        tx = Stx[n, :].copy()
        rx = Srx[n, :].copy()

        tx = tx / np.sqrt(np.mean(np.abs(tx) ** 2))
        rx = rx / np.sqrt(np.mean(np.abs(rx) ** 2))

        def fun(h):
            h_val = h[0]
            err = h_val * tx - rx
            return np.real(np.vdot(err, err))

        res = minimize(
            fun,
            x0=np.array([1.0]),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 1000},
        )

        c[n] = res.x[0]
        N0[n] = (1 - c[n] ** 2) / (c[n] ** 2)

    return N0, c


def Metric_Calculation(RxSignal, TxSignal, title):
    RxSignal = np.asarray(RxSignal).reshape(-1)
    Signal_matched_len2 = np.asarray(TxSignal).reshape(-1)[: len(RxSignal)]

    N0_OFDM, _ = getN0_MMSE(Signal_matched_len2, RxSignal)
    print(N0_OFDM)
    N0_scalar = np.asarray(N0_OFDM).reshape(-1)[0]



    SNR_OFDM = 10 * np.log10((1 - N0_scalar) / N0_scalar)
    print(f"SNR of {title} symbols is {SNR_OFDM:.6f} dB")

    return float(np.real(SNR_OFDM))



# =========================================================
# Top-level TX / RX
# =========================================================

def TX(params):
    """
    MATLAB:
        function TX_output = TX(params)
    """
    para = 256
    fftlen = 256
    nd = 40
    ml = 6
    sr = 250e3
    OFDM_bandwidth = 5e9
    AwgSampleRate = 120e9
    gilen = 32
    fftlen2 = fftlen + gilen
    pilot_amp = 0.1
    roll = 0.1
    order = 64
    sps = 8

    parabits = load_parabits_mat(params.get("parabits_path", "parabits.mat"))
    paradata = parabits.reshape(para, nd * ml, order="F")

    QAM_I, QAM_Q = qpskmod_with_qammod(paradata, para, nd, ml)
    QAM_I_norm = QAM_I / np.sqrt(42)
    QAM_Q_norm = QAM_Q / np.sqrt(42)

    Signal = QAM_I_norm + 1j * QAM_Q_norm
    Signal = Signal.reshape(para * nd, 1, order="F")

    xI, xQ = crmapping(QAM_I_norm, QAM_Q_norm, fftlen, nd)
    x = xI + 1j * xQ
    y = np.fft.ifft(x, n=fftlen, axis=0)

    ich_cp, qch_cp = giins(np.real(y), np.imag(y), fftlen, gilen, nd)
    Signal_gi = ich_cp + 1j * qch_cp
    Signal_serial = Signal_gi.reshape(fftlen2 * nd, 1, order="F")
    Signal_OFDM = Signal_serial.reshape(-1)

    Rt = params["Rt"]
    para_N1 = params["para_N1"]
    para_N2 = params["para_N2"]
    k = params["k"]

    p = np.ceil(Rt) - Rt
    q  = Rt - np.floor(Rt)
    N1 = int(np.floor(Rt) - 1)
    N2 = int(np.ceil(Rt) - 1)

    Sig_N1, Sig_N2, Signal_OFDM_Segment1, Signal_OFDM_Segment2 = Continous_DA_Mod(
        Signal_OFDM, para_N1, para_N2, Rt
    )

    Signal_N1 = Sig_N1
    Signal_N2 = Sig_N2

    Signal_original_N1 = apply_rrc_tx(Sig_N1, beta=roll, sps=sps, span=order)
    Signal_original_N2 = apply_rrc_tx(Sig_N2, beta=roll, sps=sps, span=order)

    B0 = OFDM_bandwidth
    B1 = B0 * p * (N1 + 1)
    B2 = B0 * q * (N2 + 1)

    if B1 <= B2:
        Sys_Symbolrate = B2
        Upsample_Ration = B2 / B1
        up = int(round(Upsample_Ration * sps))
        down = sps
        Signal_original_N1 = resample_poly(Signal_original_N1, up, down)
    else:
        Sys_Symbolrate = B1
        Upsample_Ration = B1 / B2
        up = int(round(Upsample_Ration * sps))
        down = sps
        Signal_original_N2 = resample_poly(Signal_original_N2, up, down)

    Sys_bandwidth = B1 + B2

    G = 2e9
    Sig_inner = Signal_original_N1
    Sig_outer = Signal_original_N2
    f_inner = B1 / 2 + G
    f_outer = f_inner + B1 / 2 + B2 / 2 + G

    Sig_inner = awgn_measured(Sig_inner, 24) #原本是23
    Sig_outer = awgn_measured(Sig_outer, 24) #原本是20，paper中是24

    L_Sig_inner = len(Sig_inner)
    L_Sig_outer = len(Sig_outer)

    L = max(L_Sig_inner, L_Sig_outer)
    Sig_inner = np.concatenate([Sig_inner, np.zeros(L - L_Sig_inner, dtype=complex)])
    Sig_outer = np.concatenate([Sig_outer, np.zeros(L - L_Sig_outer, dtype=complex)])

    PARAM_MSC = setSimulationParams(sps * Sys_Symbolrate, L)
    t = PARAM_MSC["t"]

    Sig_inner_fdm = k * Sig_inner * np.exp(1j * 2 * np.pi * f_inner * t)
    Sig_outer_fdm = Sig_outer * np.exp(1j * 2 * np.pi * f_outer * t)

    Sig_Countinous_DA = Sig_inner_fdm + Sig_outer_fdm
    RxSignal = Sig_Countinous_DA

    TX_output = {
        "RxSignal": RxSignal,
        "Signal_N1": Signal_N1,
        "Signal_N2": Signal_N2,
        "Signal_OFDM_Segment1": Signal_OFDM_Segment1,
        "Signal_OFDM_Segment2": Signal_OFDM_Segment2,
        "L_Sig_inner": L_Sig_inner,
        "L_Sig_outer": L_Sig_outer,
        "Rt": Rt,
        "para_N1": para_N1,
        "para_N2": para_N2,
        "k": k,
        "Sys_Symbolrate": Sys_Symbolrate,
        "Sys_bandwidth": Sys_bandwidth,
        "Signal_OFDM": Signal_OFDM,
    }

    return TX_output


def RX(TX_output, params):
    """
    MATLAB:
        function [SNR_OFDM_N1, SNR_OFDM_N2,SNR_total,BER] = RX(TX_output, params)
    """
    para = 256
    fftlen = 256
    nd = 40
    ml = 6
    sr = 250e3
    gilen = 32
    fftlen2 = fftlen + gilen
    OFDM_bandwidth = 5e9
    AwgSampleRate = 120e9

    roll = 0.1
    order = 64
    sps = 8

    Rt = params["Rt"]
    p = np.ceil(Rt) - Rt
    q = Rt - np.floor(Rt)
    N1 = int(np.floor(Rt) - 1)
    N2 = int(np.ceil(Rt) - 1)
    k = params["k"]

    RxSignal = TX_output["RxSignal"]
    Signal_N1 = TX_output["Signal_N1"]
    Signal_N2 = TX_output["Signal_N2"]
    Signal_OFDM_Segment1 = TX_output["Signal_OFDM_Segment1"]
    Signal_OFDM_Segment2 = TX_output["Signal_OFDM_Segment2"]
    L_Sig_inner = TX_output["L_Sig_inner"]
    L_Sig_outer = TX_output["L_Sig_outer"]
    Signal_OFDM = TX_output["Signal_OFDM"]


    B0 = OFDM_bandwidth
    B1 = B0 * p * (N1 + 1)
    B2 = B0 * q * (N2 + 1)

    if B1 <= B2:
        Sys_Symbolrate = B2
    else:
        Sys_Symbolrate = B1

    G = 2e9
    f_inner = B1 / 2 + G
    f_outer = f_inner + B1 / 2 + B2 / 2 + G

    nSamples = len(RxSignal)
    PARAM_MSC = setSimulationParams(sps * Sys_Symbolrate, nSamples)
    t = PARAM_MSC["t"]

    RX_Signal_original_N1 = (1 / k) * RxSignal * np.exp(-1j * 2 * np.pi * f_inner * t)
    RX_Signal_original_N2 = RxSignal * np.exp(-1j * 2 * np.pi * f_outer * t)

    RX_Signal_original_N1 = RX_Signal_original_N1[:L_Sig_inner]
    RX_Signal_original_N2 = RX_Signal_original_N2[:L_Sig_outer]

    if B1 <= B2:
        Upsample_Ration = B2 / B1
        up = sps
        down = int(round(Upsample_Ration * sps))
        RX_Signal_original_N1 = resample_poly(RX_Signal_original_N1, up, down)
    else:
        Upsample_Ration = B1 / B2
        up = sps
        down = int(round(Upsample_Ration * sps))
        RX_Signal_original_N2 = resample_poly(RX_Signal_original_N2, up, down)

    RX_Signal_N1 = apply_rrc_rx(RX_Signal_original_N1, beta=roll, sps=sps, span=order)
    RX_Signal_N2 = apply_rrc_rx(RX_Signal_original_N2, beta=roll, sps=sps, span=order)

    RX_Signal_N1 = RX_Signal_N1[::sps]
    RX_Signal_N2 = RX_Signal_N2[::sps]

    para_N1 = params["para_N1"]
    para_N2 = params["para_N2"]

    Rx_OFDM_Sig, RX_OFDM_Sig_Segment1, RX_OFDM_Sig_Segment2 = Continous_DA_DeMod(
        RX_Signal_N1, RX_Signal_N2, para_N1, para_N2, Rt
    )

    SNR_OFDM_N1 = Metric_Calculation(RX_OFDM_Sig_Segment1, Signal_OFDM_Segment1, "1")
    SNR_OFDM_N2 = Metric_Calculation(RX_OFDM_Sig_Segment2, Signal_OFDM_Segment2, "2")
    SNR_total = Metric_Calculation(Rx_OFDM_Sig, Signal_OFDM, "t")

    parabits = load_parabits_mat(params.get("parabits_path", "parabits.mat")).reshape(-1, 1)

    Rx_OFDM_Sig_reshape = np.asarray(Rx_OFDM_Sig).reshape(fftlen2, nd, order="F")
    _, demodata_sequence = OFDM_DeMod(
        Rx_OFDM_Sig_reshape, fftlen2, gilen, nd, fftlen, para, ml
    )

    RxSignal_after_LMS = demodata_sequence.reshape(-1, 1)
    ref_signal_bit = parabits.reshape(-1, 1)

    min_len = min(len(RxSignal_after_LMS), len(ref_signal_bit))
    error_bit_after_LMS = np.sum(RxSignal_after_LMS[:min_len] != ref_signal_bit[:min_len])
    BER = error_bit_after_LMS / min_len

    print(f"BER = {BER:.6e}")

    return SNR_OFDM_N1, SNR_OFDM_N2, SNR_total, BER
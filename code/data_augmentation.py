# Data augmentation for EMG signals adapted from:
#       T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks,” in Proceedings of the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 2017. New York, NY, USA: ACM, 2017, pp. 216–220. 
#       https://dl.acm.org/citation.cfm?id=3136817
#       https://arxiv.org/abs/1706.00527
#       @inproceedings{TerryUm_ICMI2017,
#           author = {Um, Terry T. and Pfister, Franz M. J. and Pichler, Daniel and Endo, Satoshi and Lang, Muriel and Hirche, Sandra and Fietzek, Urban and Kuli\'{c}, Dana},
#           title = {Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks},
#           booktitle = {Proceedings of the 19th ACM International Conference on Multimodal Interaction},
#           series = {ICMI 2017},
#           year = {2017},
#           isbn = {978-1-4503-5543-8},
#           location = {Glasgow, UK},
#           pages = {216--220},
#           numpages = {5},
#           doi = {10.1145/3136755.3136817},
#           acmid = {3136817},
#           publisher = {ACM},
#           address = {New York, NY, USA},
#           keywords = {Parkinson\&\#39;s disease, convolutional neural networks, data augmentation, health monitoring, motor state detection, wearable sensor},
#       }


import numpy as np
from scipy.interpolate import CubicSpline
from scipy import signal


def jitter(x, snr_db=25):
    if isinstance(snr_db, list):
        snr_db_low = snr_db[0]
        snr_db_up = snr_db[1]
    else:
        snr_db_low = snr_db
        snr_db_up = 45
    snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]
    snr = 10 ** (snr_db / 10)
    Xp = np.sum(x**2, axis=0, keepdims=True) / x.shape[0]
    Np = Xp / snr
    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)
    xn = x + n
    return xn


def scale(x, sigma=0.2):
    '''Multiply signal with random scalar from normal distribution N(1,sigma).'''
    a = np.random.normal(size=x.shape[1], scale=sigma, loc=1.0)
    output = a * x
    return output


def rotate(x, rotation=2, mask=None):
    '''Rotate signal channels randomly between [0,2] positions. Use mask to disable rotation of specific channels'''
    output = np.zeros(x.shape)
    r = np.random.randint(-rotation, rotation + 1, size=1)[0]
    if mask is None:
        mask = np.ones(x.shape[1])
    channels = np.arange(x.shape[1])
    rolled = np.roll(channels[np.where(mask)], r)
    channels[np.where(mask)] = rolled
    output = x[:, channels]
    return output


def _generate_random_curve(x, sigma=0.2, knot=4):
    xx = ((np.arange(0, x.shape[0], (x.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, ))
    x_range = np.arange(x.shape[0])
    cs = CubicSpline(xx[:], yy[:])
    return np.array(cs(x_range)).transpose()


def _distort_timesteps(x, sigma=0.2):
    # Regard these samples aroun 1 as time intervals
    tt = _generate_random_curve(x, sigma)
    # Add intervals to make a cumulative graph
    tt_cum = np.cumsum(tt, axis=0)
    # Make the last value to have X.shape[0]
    t_scale = (x.shape[0] - 1) / tt_cum[-1]
    tt_cum = tt_cum * t_scale
    return tt_cum


def mag_warp(x, sigma=0.2):
    output = np.zeros(x.shape)
    for i in range(x.shape[1]):
        rc = _generate_random_curve(x[:, i], sigma)
        output[:, i] = x[:, i] * rc
    return output


def time_warp(x, sigma=0.1):
    output = np.zeros(x.shape)
    for i in range(x.shape[1]):
        tt_new = _distort_timesteps(x[:, i], sigma)
        tt_new = np.clip(tt_new, 0, x.shape[0] - 1)
        output[:, i] = x[tt_new.astype(int), i]
    return output


def permute(x, nPerm=4, minSegLength=10):
    def max_seg(segments):
        m = 0
        for i in range(len(segments) - 1):
            if (segments[i + 1] - segments[i]) >= (segments[m + 1] - segments[m]):
                m = i
        return (segments[m], segments[m + 1])
    segs = [0, x.shape[0]]

    it = 0
    while len(segs) < nPerm + 1:
        a, b = max_seg(segs)
        if b - a > 2 * minSegLength:
            p = np.random.randint(a, b, size=1)[0]
            if ((b - p) >= minSegLength) and ((p - a) >= minSegLength):
                segs.append(p)
        elif b - a == 2 * minSegLength:
            segs.append((b + a) / 2)
        else:
            break
        segs.sort()
        it += 1
    segs = np.array(segs, dtype=int)
    idx = np.arange(len(segs) - 1)
    np.random.shuffle(idx)
    output = np.zeros(x.shape)
    pp = 0
    for ii in range(len(idx)):
        x_temp = x[segs[idx[ii]]:segs[idx[ii] + 1], :]
        output[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return output

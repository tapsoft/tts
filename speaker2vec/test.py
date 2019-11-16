import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from trim import trim
from python_speech_features import mfcc


def get_feature(filepath):
    # return mfcc feature as a numpy array with shape (n_mfcc, t)
    print("file: " + filepath[-22:])

    # load audio file
    (rate, sig) = wav.read(filepath)
    sig = sig.ravel()
    print('loaded, length %d' % sig.shape[0])

    # trim silence
    sigt = trim(sig)
    print('trimmed, length %d' % sigt.shape[0])

    # extract mfcc features
    # 40 mel-space filters, 25ms hamming window, 10ms shift
    feat = mfcc(signal=sigt, samplerate=rate, winlen=0.025, winstep=0.01, numcep=40, nfilt=40).T
    # feat = np.random.randn(40, 400)
    print("feature obtained, shape (%d, %d)" % (feat.shape[0], feat.shape[1]))


    plt.subplot(4, 1, 1)
    plt.plot(sig)
    plt.subplot(4, 1, 2)
    plt.plot(sigt)
    plt.subplot(4, 1, 3)
    plt.imshow(feat, cmap='jet')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.subplot(4, 1, 4)
    plt.imshow(feat[:, 0:200], cmap='jet')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
    print(feat[:, 0:200].shape)

    del sig, sigt
    return feat


get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000001.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000011.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000777.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000111.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_621857.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_269097.wav")

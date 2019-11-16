
import librosa
import numpy as np
import matplotlib.pyplot as plt


def get_feature(filepath, sr=16000, train_mode=False):
    # load audio file
    # return mfcc feature as a numpy array with shape (n_mfcc, t)
    y, _ = librosa.load(filepath, mono=True, sr=sr)
    yt, idx = librosa.effects.trim(y, top_db=25)
    # extract mfcc features
    # 40 mel-space filters, 25ms hamming window, 10ms shift
    feat = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=40, hop_length=int(sr/100), n_fft=int(sr/40))

    plt.subplot(4, 1, 1)
    plt.plot(y)
    plt.subplot(4, 1, 2)
    plt.plot(yt)
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

    del y, yt
    return feat


get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000001.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000011.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000777.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000111.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000999.wav")

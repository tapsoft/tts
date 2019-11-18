import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from trim import trim
from python_speech_features import mfcc
import torch.nn as nn
import torch.optim as optim
from loader import *
import queue
from AutoEncoder import AutoEncoder
from main import FILE_PATHS, SAVE_PATH, n_mfcc, n_frames, learning_rate, num_workers, import_paths, load
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

######################################################################################################

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

"""
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000001.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000011.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000777.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_0001/KsponSpeech_000111.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_621857.wav")
get_feature("D:/GitHub_Repos/zeroshot-tts-korean/data_sample/KsponSpeech_269097.wav")
"""
######################################################################################################

file_paths = []
num_samples = 100


def load(filename, model, optimizer):
    state = torch.load(filename)
    model.load_state_dict(state['model'])
    if 'optimizer' in state and optimizer:
        optimizer.load_state_dict(state['optimizer'])
    logger.info('checkpoint loaded from ' + filename)
    return state['best_loss']


# set random seed
seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# set device
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

# import data file paths
f = open(FILE_PATHS, "r")
lines = f.readlines()
for line in lines:
    file_paths.append(line[:-1])
f.close()

for i in range(10):
    logger.debug(file_paths[i])
logger.debug("...")
# randomly shuffle data
random.shuffle(file_paths)

# initialize model
model = AutoEncoder(d=n_mfcc * n_frames)
logger.info("model: " + str(model))

# load model to device
model = nn.DataParallel(model).to(device)

# set optimizer and loss
optimizer = optim.Adam(model.module.parameters(), lr=learning_rate)

# load pre-trained model
try:
    logger.info('loading checkpoint...')
    loaded_loss = load(SAVE_PATH + "best_eval.pt", model, optimizer)
except:
    raise RuntimeError('no checkpoint loaded')
else:
    logger.info('checkpoint loaded; best loss %0.4f' % loaded_loss)

# set model to eval mode
model.eval()

# vis_dataset: BaseDataset object of visualization data
vis_dataset = BaseDataset(file_paths[0:num_samples], train_mode=False)

vis_queue = queue.Queue(num_workers * 2)
vis_loader = BaseDataLoader(vis_dataset, vis_queue, num_samples, 0)
vis_loader.start()

# begin logging
logger.info('visualize start')
begin = time.time()

with torch.no_grad():
    # input, target tensor shapes: (batch_size, n_mfcc, n_frames)
    inputs, _ = vis_queue.get()
    batch_size = inputs.shape[0]

    # load tensors to device
    inputs = inputs.to(device)

    # output tensor shape: (batch_size, n_mfcc, n_frames)
    # forward pass
    # compressed features as a numpy array of (batch_size, hidden_size)
    embedding = model(inputs).cpu().numpy()

vis_loader.join()

pca = PCA(n_components=2)
embedding = np.array(embedding.reshape((embedding.shape[0], -1)))
pca_result = pca.fit_transform(embedding)
pca_1 = pca_result[:, 0]
pca_2 = pca_result[:, 1]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

fig = plt.figure(figsize=(16, 10))
plt.scatter(x=np.squeeze(pca_1), y=np.squeeze(pca_2))
plt.title("PCA")
plt.xlabel("component 1")
plt.ylabel("component 2")
fig.savefig('./PCA.png')

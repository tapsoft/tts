from encoder import inference as encoder
from pathlib import Path
import numpy as np
import librosa
import random
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


encoder_weights = Path('./encoder/pretrained.pt')
encoder.load_model(encoder_weights)

file_paths = []

# import data file paths
f = open('./encoder/file_paths.txt', "r")
lines = f.readlines()
for line in lines:
    file_paths.append(line[:-1])
f.close()

maxi = 1000  # number of embeddings to generate

embeddings = []
for i, file_path in enumerate(file_paths):
    print("file %d/%d import" % (i+1, maxi))
    in_fpath = Path(file_path)

    # load and preprocess audio
    original_wav, sampling_rate = librosa.load(in_fpath, sr=16000)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)

    # make speaker embedding
    embed = encoder.embed_utterance(preprocessed_wav)
    embeddings.append(embed)

    if i == maxi:
        break

# randomly shuffle data
random.shuffle(file_paths)

print("load complete")

embeddings = np.stack(embeddings, axis=0)

# PCA analysis
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)
pca_1 = pca_result[:, 0]
pca_2 = pca_result[:, 1]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

fig = plt.figure(figsize=(16, 16))
plt.scatter(x=np.squeeze(pca_1), y=np.squeeze(pca_2))
plt.title("PCA")
plt.xlabel("component 1")
plt.ylabel("component 2")
fig.savefig('./PCA.png')

print("PCA complete")

# t-SNE analysis
tsne = TSNE(random_state=0)
tsne_result = tsne.fit_transform(embeddings)
tsne_1 = tsne_result[:, 0]
tsne_2 = tsne_result[:, 1]

fig = plt.figure(figsize=(16, 16))
plt.scatter(x=np.squeeze(tsne_1), y=np.squeeze(tsne_2))
plt.title("t-SNE")
plt.xlabel("component 1")
plt.ylabel("component 2")
fig.savefig('./TSNE.png')

print("t-SNE complete")

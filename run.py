import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from tacotron2.hparams import create_hparams
from tacotron2.train import load_model
from tacotron2.utils import load_wav_to_torch, load_filepaths_and_text
from tacotron2.speaker_embed.encoder import inference as encoder
from tacotron2.text import text_to_sequence
import tacotron2.layers as layers
from vocoder.vocoder import generate as vocoder_generate


def get_mel_from_wavfile(hparams, filename):
    stft = layers.TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)

    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    mel_spec = stft.mel_spectrogram(audio_norm)
    # mel_spec = torch.squeeze(mel_spec, 0)

    return mel_spec, sampling_rate

    
def get_mel_from_tacotron2(audiopath, text):

    audio, sampling_rate = load_wav_to_torch(audiopath)
    audio = audio.numpy()

    preprocessed_wav = encoder.preprocess_wav(audio, sampling_rate)
    embed = encoder.embed_utterance(preprocessed_wav)
    embed = torch.Tensor(embed).cuda()

    sequence = np.array(text_to_sequence(text))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, embed)

    return mel_outputs, mel_outputs_postnet, alignments


def mel_rescale(mel):
    # Normalize and rescale mel outputs
    min_value = torch.min(mel)
    max_value = torch.max(mel)
    mel = (mel - min_value) / (max_value - min_value)
    vocoder_input = mel * 8. - 4.
    return vocoder_input


def get_mel_image(mel, imagetitle='Mel-Spectrogram', filename='melspec.png'):
    fig = plt.figure()
    plt.imshow(mel.squeeze().cpu().detach().numpy())
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(imagetitle)
    plt.xlabel('encoder timestep')
    plt.ylabel('decoder timestep')
    fig.savefig(filename)



if __name__ == '__main__':

    # Inputs
    checkpoint_path = "pretrained/checkpoint_41000"
    audiopath = 'input/input_voice.wav'
    text = '가나다라마바사아자차카타파하'

    # Load synthesizer model (speaker embedding + tacotron2 model) and hparams
    hparams = create_hparams()
    model = load_model(hparams)

    # Load checkpoint
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, checkpoint_dict['iteration']))

    # Set model to evaluation mode
    _ = model.cuda().eval()

    # Get mel_spec directly from wav file
    # Also retrieve original sampling rate
    mel, sr = get_mel_from_wavfile(hparams, audiopath)

    # Get mel spectrogram from synthesizer
    mel_outputs, mel_outputs_postnet, alignments = get_mel_from_tacotron2(audiopath, text)

    # Rescale amplitude range to -4 ~ +4
    mel_from_wav = mel_rescale(mel)
    mel_from_tacotron2 = mel_rescale(mel_outputs)
    mel_from_tacotron2_postnet = mel_rescale(mel_outputs_postnet)

    # Save mel-spectrogram image file
    get_mel_image(mel_from_wav, filename='output/mel_wav.png', imagetitle='mel_wav')
    get_mel_image(mel_from_tacotron2, filename='output/mel_tacotron2.png', imagetitle='mel_tacotron2')
    get_mel_image(mel_from_tacotron2_postnet, filename='output/mel_tacotron2_postnet.png', imagetitle='mel_tacotron2_postnet')
    get_mel_image(alignments, filename='output/alignments.png', imagetitle='alignments')

    # Generate wav file using vocoder
    # Restore sampling rate before exporting to .wav file
    vocoder_generate(mel_from_wav, 'mel_from_wav.wav', sampling_rate=sr)
    vocoder_generate(mel_from_tacotron2, 'mel_from_tacotron2.wav', sampling_rate=sr)
    vocoder_generate(mel_from_tacotron2_postnet, 'mel_from_tacotron2_postnet.wav', sampling_rate=sr)

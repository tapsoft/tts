import torch
import argparse
import zipfile, os

from vocoder.models.fatchord_version import WaveRNN
from vocoder.utils import hparams as hp
from vocoder.utils.text.symbols import symbols
from vocoder.models.tacotron import Tacotron
from vocoder.utils.text import text_to_sequence
from vocoder.utils.display import save_attention, simple_table

def generate(mel):
    os.makedirs('vocoder/pretrained/voc_weights/', exist_ok=True)

    zip_ref = zipfile.ZipFile('vocoder/pretrained/ljspeech.wavernn.mol.800k.zip', 'r')
    zip_ref.extractall('vocoder/pretrained/voc_weights/')
    zip_ref.close()

    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation (lower quality)')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slower Unbatched Generation (better quality)')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='vocoder/hparams.py',
                        help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Using device:', device)

    print('\nInitialising WaveRNN Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode='MOL').to(device)

    voc_model.load('vocoder/pretrained/voc_weights/latest_weights.pyt')

    save_path = f'output.wav'

    # save_attention(attention, save_path)

    mel = torch.tensor(mel).unsqueeze(0)
    mel = (mel + 4) / 8
    voc_model.generate(mel, save_path, False, 11_000, 550, hp.mu_law)

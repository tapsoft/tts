import time
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from vocoder.utils.display import stream, simple_table
from vocoder.utils.dataset import get_vocoder_datasets
from vocoder.utils.distribution import discretized_mix_logistic_loss
from vocoder.utils import hparams as hp
from vocoder.models.fatchord_version import WaveRNN
from vocoder.gen_wavernn import gen_testset
from vocoder.utils.paths import Paths
import argparse
from vocoder.utils import data_parallel_workaround
from vocoder.utils.checkpoints import save_checkpoint, restore_checkpoint

def main():

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train WaveRNN Vocoder')
    parser.add_argument('--lr', '-l', type=float,  help='[float] override hparams.py learning rate')
    parser.add_argument('--batch_size', '-b', type=int, help='[int] override hparams.py batch size')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--gta', '-g', action='store_true', help='train wavernn on GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.training_files = "tacotron2/filelists/transcripts_korean_final_final.txt"
    hp.validation_files = "tacotron2/filelists/transcripts_korean_final_validate.txt"
    hp.filter_length = 1024
    hp.n_mel_channels = 80
    hp.sampling_rate = 16000
    hp.mel_fmin = 0.0
    hp.mel_fmax = 8000.0
    hp.max_wav_value = 32768.0
    hp.n_frames_per_step = 1
    # hp.data_path = "../data/"

    # hp.win_length = 1024


    hp.configure(args.hp_file)  # load hparams from file
    if args.lr is None:
        args.lr = hp.voc_lr
    if args.batch_size is None:
        args.batch_size = hp.voc_batch_size

    paths = Paths("../data/", hp.voc_model_id, hp.tts_model_id)
    # paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    batch_size = 64
    force_train = args.force_train
    train_gta = args.gta
    lr = args.lr

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        if batch_size % torch.cuda.device_count() != 0:
            raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising Model...\n')

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
                        mode=hp.voc_mode).to(device)

    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    optimizer = optim.Adam(voc_model.parameters())
    restore_checkpoint('voc', paths, voc_model, optimizer, create_if_missing=True)

    train_set, test_set = get_vocoder_datasets(paths.data, batch_size, train_gta, hp)

    total_steps = 10_000_000 if force_train else hp.voc_total_steps

    simple_table([('Remaining', str((total_steps - voc_model.get_step())//1000) + 'k Steps'),
                  ('Batch Size', batch_size),
                  ('LR', lr),
                  ('Sequence Len', hp.voc_seq_len),
                  ('GTA Train', train_gta)])

    loss_func = F.cross_entropy if voc_model.mode == 'RAW' else discretized_mix_logistic_loss

    voc_train_loop(paths, voc_model, loss_func, optimizer, train_set, test_set, lr, total_steps)

    print('Training Complete.')
    print('To continue training increase voc_total_steps in hparams.py or use --force_train')


from tacotron2.utils import load_filepaths_and_text, load_wav_to_torch
import random
# from run import get_mel_from_wavfile
import tacotron2.layers as layers

def get_mel_from_wavfile(hparams, filename):
    # print("filter, window", hparams.filter_length, hparams.win_length)
    # print("filename", filename)
    stft = layers.TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.filter_length,
        # hparams.filter_length, hparams.hop_length, hparams.win_length,
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

    return mel_spec


from torch.utils.data.dataloader import DataLoader
from vocoder.utils.dataset import collate_vocoder
from vocoder.utils.dataset import VocoderDataset

def get_vocoder_datasets(path: Paths, batch_size, train_gta, hparams):
    print("path", path)
    # train_dataset = MelLoader(hparams.training_files, hparams)
    # test_dataset = MelLoader(hparams.validation_files, hparams)
    # collate_fn = MelCollate(hparams.n_frames_per_step)

    with open('tacotron2/filelists/transcripts_korean_final_final.txt', encoding='utf-8') as f:
        train_ids = [line.strip().split("|")[0].split("/")[-1][:-4] for line in f]
        # print("data", dataset[:3])
    with open('tacotron2/filelists/transcripts_korean_final_validate.txt', encoding='utf-8') as f:
        test_ids = [line.strip().split("|")[0].split("/")[-1][:-4] for line in f]
    dataset_rate = 0.1
    train_ids = train_ids[:int(len(train_ids) * dataset_rate)]
    test_ids = test_ids[:int(len(test_ids) * dataset_rate)]
    print("batch size", batch_size)
    # dataset_ids = [x[0] for x in dataset]


    # random.seed(1234)
    # random.shuffle(dataset_ids)

    # test_ids = dataset_ids[-hp.voc_test_samples:]
    # train_ids = dataset_ids[:-hp.voc_test_samples]

    train_dataset = VocoderDataset(path, train_ids, train_gta)
    test_dataset = VocoderDataset(path, test_ids, train_gta)

    train_set = DataLoader(train_dataset,
                           collate_fn=collate_vocoder,
                           batch_size=batch_size,
                           num_workers=8,
                           shuffle=True,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                          batch_size=1,
                          num_workers=8,
                          shuffle=False,
                          pin_memory=True)

    return train_set, test_set

def voc_train_loop(paths: Paths, model: WaveRNN, loss_func, optimizer, train_set, test_set, lr, total_steps):
    # Use same device as model parameters
    device = next(model.parameters()).device

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = (total_steps - model.get_step()) // total_iters + 1

    for e in range(1, epochs + 1):

        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(train_set, 1):
            # print("size", x.size())
            x, m, y = x.to(device), m.to(device), y.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                y_hat = data_parallel_workaround(model, x, m)
            else:
                # print("size", x.size(), m.size())
                y_hat = model(x, m)

            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)

            elif model.mode == 'MOL':
                y = y.float()

            y = y.unsqueeze(-1)


            loss = loss_func(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            if hp.voc_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.voc_clip_grad_norm)
                if np.isnan(grad_norm):
                    print('grad_norm was NaN!')
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.voc_checkpoint_every == 0:
                gen_testset(model, test_set, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                            hp.voc_target, hp.voc_overlap, paths.voc_output)
                ckpt_name = f'wave_step{k}K'
                save_checkpoint('voc', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:.4f} | {speed:.1f} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('voc', paths, model, optimizer, is_silent=True)
        model.log(paths.voc_log, msg)
        print(' ')


if __name__ == "__main__":
    main()
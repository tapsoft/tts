# Team 18 CS470 Final Project
## Transfer Learning from Speaker Verification to Zero-Shot Multispeaker Korean Text-To-Speech Synthesis

Proposal link: https://docs.google.com/document/d/1x8I7riwTYUAlwgf8Ou8jAZ57uNL79e8VIzXJ8rxSlEg/

Korean multispeaker speech dataset: http://www.aihub.or.kr/content/552

## Inference

1. Install required packages by running following command on linux terminal.
```
pip install -r requirements.txt
```

2. Place input text ```input_text.txt``` and reference voice ```input_voice.wav``` in ```./input``` folder.

3. Run ```run.py``` on python interpreter.
```
python3 ./run.py
```

4. Find generated speech at ```./output/generated.wav```.

## Repository documentation

```./input```: Contains user query files (input text / reference voice)

```./output```: Output stored

```./past```: [Not used] Contains autoencoder-based, unsupervised speaker embedding generator. As RNN-based embedder pretrained on English dataset turned out to perform better than this model, we are not using this.

```./tacotron2```: Directory for tacotron2-based text-to-mel spectrogram synthesizer. Contains speaker embedding generator at ```./tacotron2/speaker_embed```, pretrained on English speaker verification task.

```./vocoder```: Directory for WaveNet-based mel-to-audio vocoder.

## Tacotron2-based synthesizer

## WaveNet-based vocoder
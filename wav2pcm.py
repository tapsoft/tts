
import os
import time
import wave

FILE_PATHS = "./speaker2vec/file_paths.txt"
file_paths = []
channels = 1
bit_depth = 16
sampling_rate = 16000


def make_paths():
    print("constructing file paths")
    # list of audio data paths
    for i in range(4):
        base_folder = '/data/KsponSpeech_0' + str(i + 1)
        for j in range(124):
            folder_index = str(124 * i + j + 1)
            folder = base_folder + '/KsponSpeech_' + '0' * (4 - len(folder_index)) + folder_index
            for k in range(1000):
                file_index = str((124 * i + j) * 1000 + k + 1)
                filename = '/KsponSpeech_' + '0' * (6 - len(file_index)) + file_index + '.pcm'
                file_paths.append(folder + filename)

    for j in range(497, 623):
        index = str(j)
        folder = '/data/KsponSpeech_05/KsponSpeech_0' + str(j)
        for k in range(1000):
            file_index = str((j - 1) * 1000 + k + 1)
            filename = '/KsponSpeech_' + file_index + '.pcm'
            file_paths.append(folder + filename)

    for k in range(622001, 622545):
        file_index = str(k)
        folder = '/data/KsponSpeech_05/KsponSpeech_0623'
        filename = '/KsponSpeech_' + file_index + '.pcm'
        file_paths.append(folder + filename)

    f = open(FILE_PATHS, "w")
    for file_path in file_paths:
        f.write(file_path+'\n')
        f.close()

    return None


def import_paths():
    print("loading file paths")

    f = open(FILE_PATHS, "r")
    lines = f.readlines()
    for line in lines:
        file_paths.append(line[:-1])
    f.close()

    return None


def main():
    # import audio file paths
    if not os.path.isfile(FILE_PATHS):
        make_paths()
    else:
        import_paths()
    for i in range(100):
        print(file_paths[i])

    check = time.time()
    # convert each .pcm files to .wav files and save
    for i, file_path in enumerate(file_paths):
        if i % 100 == 1:
            print("processing " + file_path)
            print("mean time per file: " + str((time.time()-check)/100) + " sec")
            check = time.time()

        with open(file_path, "rb") as opened_pcm_file:
            pcm_data = opened_pcm_file.read()
            wavefile = file_path[:-3] + "wav"
            obj2write = wave.open(wavefile, "wb")
            obj2write.setnchannels(channels)
            obj2write.setsampwidth(bit_depth // 8)
            obj2write.setframerate(sampling_rate)
            obj2write.writeframes(pcm_data)
            obj2write.close()


if __name__ == "__main__":
    main()


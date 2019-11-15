
FILE_PATHS = "./file_paths.txt"
file_paths = []


def make_paths():
    print("constructing file paths")
    # list of audio data paths
    for i in range(4):
        base_folder = '/data/KsponSpeech_wav'
        for j in range(124):
            folder_index = str(124 * i + j + 1)
            folder = base_folder + '/KsponSpeech_' + '0' * (4 - len(folder_index)) + folder_index
            for k in range(1000):
                file_index = str((124 * i + j) * 1000 + k + 1)
                filename = '/KsponSpeech_' + '0' * (6 - len(file_index)) + file_index + '.wav'
                file_paths.append(folder + filename)

    for j in range(497, 623):
        index = str(j)
        folder = '/data/KsponSpeech_wav/KsponSpeech_0' + str(j)
        for k in range(1000):
            file_index = str((j - 1) * 1000 + k + 1)
            filename = '/KsponSpeech_' + file_index + '.wav'
            file_paths.append(folder + filename)

    for k in range(622001, 622545+1):
        file_index = str(k)
        folder = '/data/KsponSpeech_wav/KsponSpeech_0623'
        filename = '/KsponSpeech_' + file_index + '.wav'
        file_paths.append(folder + filename)

    for i in range(100):
        print(file_paths[i])

    f = open(FILE_PATHS, "w")
    for file_path in file_paths:
        f.write(file_path+'\n')
    f.close()

    return None


# import audio file paths
make_paths()


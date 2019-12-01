def is_english_word(line, index):
    w = line[index]
    if w in ['n', 'b', 'l', 'o', 'u']:
        if line[index + 1] == '/':
            return False

    if ord('a') <= ord(w.lower()) <= ord('z'):
        return True

    return False


def preprocess(text):
    for symb in ['n', 'b', 'l', 'o', 'u']:
        text = text.replace(symb + '/ ', '')
        text = text.replace(symb + '/', '')

    for sc in ['+', '*', '/']:
        text = text.replace(sc, '')

    while text.find(')(') != -1:
        index = text.find(')(')
        prev = index - 1
        while text[prev] != '(':
            prev = prev - 1
        text = text[:prev] + text[index + 2:]

    text = text.replace(')', '')
    text = text.rstrip()

    return text


def extract_english_word(text):
    english_words = []

    length = len(text)
    i = 0
    while i < length:
        w = text[i]
        if ord('a') <= ord(w.lower()) <= ord('z'):
            word = w
            i = i + 1
            while i < length and ord('a') <= ord(text[i].lower()) <= ord('z'):
                word += text[i]
                i = i + 1
            english_words.append(word)
        i = i + 1

    return english_words


def check_file_text(filename):
    try:
        # f = open(filename, 'r', encoding='euc-kr')
        # line = f.readline()
        # processed_line = preprocess(line)
        # english_words = extract_english_word(processed_line)
        # f.close()
        with open(filename, 'r', encoding='euc-kr') as f:
            line = f.readline()
            processed_line = preprocess(line)
            english_words = extract_english_word(processed_line)
        return ['parsed', processed_line, english_words]
    except Exception as ex:
        # print('Error on ' + filename, ex)
        return ['error', filename, ex]


if __name__ == '__main__':
    transcripts_filename = 'transcripts2.txt'
    english_words_filename = 'english_words2.txt'
    error_files_filename = 'error_files2.txt'

    # Set file path
    filepaths = []

    # """
    # Full data
    for i in range(4):
        path = []
        baseFolder = '/data/KsponSpeech_0' + str(i + 1)
        for j in range(124):
            folderIndex = str(124 * i + j + 1)
            folder = baseFolder + '/KsponSpeech_' + '0' * (4 - len(folderIndex)) + folderIndex
            for k in range(1000):
                fileIndex = str((124 * i + j) * 1000 + k + 1)
                filename = 'KsponSpeech_' + '0' * (6 - len(fileIndex)) + fileIndex + '.txt'
                path.append([folder, filename])
        filepaths.append(path)

    _path = []
    for j in range(497, 623):
        index = str(j)
        folder = '/data/KsponSpeech_05/KsponSpeech_0' + str(j)
        for k in range(1000):
            fileIndex = str((j - 1) * 1000 + k + 1)
            filename = 'KsponSpeech_' + fileIndex + '.txt'
            _path.append([folder, filename])
    for k in range(622001, 622545):
        fileIndex = str(k)
        folder = '/data/KsponSpeech_05/KsponSpeech_0623'
        filename = 'KsponSpeech_' + fileIndex + '.txt'
        _path.append([folder, filename])
    filepaths.append(_path)
    # """

    """
    # Sample data
    path = []
    folder = 'data_sample/KsponSpeech_01/KsponSpeech_0001'
    for i in range(1000):
        index = str(i + 1)
        filename = 'KsponSpeech_' + '0' * (6 - len(index)) + index + '.txt'
        path.append([folder, filename])
    filepaths.append(path)
    # """

    # Read Data
    transcripts = []
    total_english_words = []
    error_files = []
    for i in range(5):
        for file in filepaths[i]:
            folder, filename = file[0], file[1]
            result = check_file_text(folder + '/' + filename)

            if result[0] == 'parsed':
                processed_line, english_words = result[1], result[2]
                # transcripts.append([filename, processed_line])
                transcripts.append([folder + '/' + filename.replace('.txt', '.pcm'), processed_line])
                if len(english_words) > 0:
                    total_english_words.append([filename, english_words])
            else:
                filename, ex = result[1], result[2]
                error_files.append([filename, ex])
        print('Reading folder ' + str(i + 1) + ' is finished')

    # Write Preprocessed Data
    f_transcripts = open(transcripts_filename, 'w')
    for element in transcripts:
        filename, processedLine = element[0], element[1]
        f_transcripts.write(filename + '|' + processedLine + '\n')
    f_transcripts.close()

    f_english_words = open(english_words_filename, 'w')
    for element in total_english_words:
        filename, english_words = element[0], element[1]
        words = ''
        for word in english_words:
            words = words + word + ', '
        f_english_words.write(filename + ': ' + words + '\n')
    f_english_words.close()

    f_error_files = open(error_files_filename, 'w')
    for element in error_files:
        filename, ex = element[0], element[1]
        f_error_files.write(filename + ': ' + str(ex) + '\n')
    f_error_files.close()

    print('Finished')

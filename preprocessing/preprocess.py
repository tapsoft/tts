def is_english_word(line, index):
    w = line[index]
    if w in ['n', 'b', 'l', 'o', 'u']:
        if line[index + 1] == '/':
            return False

    if ord('a') <= ord(w.lower()) <= ord('z'):
        return True

    return False


def preprocess_text(text):
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
        with open(filename, 'r', encoding='euc-kr') as f:
            line = f.readline()
            processed_line = preprocess_text(line)
            english_words = extract_english_word(processed_line)
        return ['parsed', processed_line, english_words]
    except Exception as ex:
        return ['error', filename, ex]


def change_word(word_from, word_to, filename_from, filename_to):
    with open(filename_from, 'r') as f:
        new_lines = []
        lines = f.readlines()
        for line in lines:
            [path, text] = line.split('|')
            length = len(text)
            index = text.find(word_from.lower())
            if index == -1:
                pass
            elif index == 0:
                if ord('a') <= ord(text[index + len(word_from)].lower()) < ord('z'):
                    pass
            elif index + len(word_from) >= length:
                if index > 1 and ord('a') <= ord(text[index - 1].lower()) < ord('z'):
                    pass
            else:
                if ord('a') <= ord(text[index - 1].lower()) < ord('z') or ord('a') <= ord(text[index + len(word_from)].lower()) < ord('z'):
                    pass
                else:
                    text = text.replace(word_from, word_to)

            index2 = text.find(word_from.upper())
            if index2 == -1:
                pass
            elif index2 == 0:
                if ord('a') <= ord(text[index2 + len(word_from)].lower()) < ord('z'):
                    pass
            elif index2 + len(word_from) >= length:
                if index2 > 1 and ord('a') <= ord(text[index2 - 1].lower()) < ord('z'):
                    pass
            else:
                if ord('a') <= ord(text[index2 - 1].lower()) < ord('z') or ord('a') <= ord(text[index2 + len(word_from)].lower()) < ord('z'):
                    pass
                else:
                    text = text.replace(word_from.upper(), word_to)

            new_lines.append(path + '|' + text)

        with open(filename_to, 'w') as g:
            print(len(new_lines))
            for line in new_lines:
                g.write(line)


def make_vector(transcripts):
    vector = []
    with open(transcripts, 'r') as f:
        lines = f.readlines()
        for line in lines:
            [_, text] = line.split('|')
            text = text.rstrip()
            length = len(text)
            for i in range(length):
                w = text[i]
                if w not in vector:
                    vector.append(w)
    return vector


def write_vector_file(vector, filename):
    num_elements = len(vector)
    vector_str = ''
    for i in range(num_elements - 1):
        vector_str = vector_str + vector[i] + '|'
    vector_str = vector_str + vector[num_elements - 1]

    with open(filename, 'w') as f:
        f.write(vector_str)

def read_vector_file(filename):
    with open(filename, 'r') as f:
        line = f.readline()
        vector = line.split('|')
    return vector

def remove_english_word(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        lines = f.readlines()
        koreans = []
        for line in lines:
            is_only_korean = True
            index = line.find('|')
            text = line[index + 1:]
            length = len(text)
            for i in range(length):
                if ord('a') <= ord(text[i].lower()) <= ord('z'):
                    is_only_korean = False
                    break
            if is_only_korean:
                koreans.append(line)
        with open(output_filename, 'w') as g:
            for line in koreans:
                g.write(line)

def make_raw_transcript():
    transcripts_filename = 'transcripts.txt'
    english_words_filename = 'english_words.txt'
    error_files_filename = 'error_files.txt'

    # Set file path
    filepaths = []

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

def modify_raw_transcript():
    # Change frequently used english word
    change_word('pc', '피씨', 'transcripts.txt', 'transcripts_pc.txt')
    change_word('tv', '티비', 'transcripts_pc.txt', 'transcripts_tv.txt')
    change_word('b', '비', 'transcripts_tv.txt', 'transcripts_b.txt')
    change_word('c', '씨', 'transcripts_b.txt', 'transcripts_c.txt')
    change_word('lg', '엘지', 'transcripts_c.txt', 'transcripts_lg.txt')
    change_word('sns', '에스엔에스', 'transcripts_lg.txt', 'transcripts_sns.txt')
    change_word('cgv', '씨지비', 'transcripts_sns.txt', 'transcripts_cgv.txt')
    change_word('pt', '피티', 'transcripts_cgv.txt', 'transcripts_pt.txt')
    change_word('mt', '엠티', 'transcripts_pt.txt', 'transcripts_mt.txt')
    change_word('ot', '오티', 'transcripts_mt.txt', 'transcripts_ot.txt')
    change_word('asmr', '에이에스엠알', 'transcripts_ot.txt', 'transcripts_asmr.txt')
    change_word('ktx', '케이티엑스', 'transcripts_asmr.txt', 'transcripts_ktx.txt')
    change_word('d', '디', 'transcripts_ktx.txt', 'transcripts_d.txt')
    change_word('kt', '케이티', 'transcripts_d.txt', 'transcripts_kt.txt')
    change_word('sk', '에스케이', 'transcripts_kt.txt', 'transcripts_sk.txt')
    change_word('dc', '디씨', 'transcripts_sk.txt', 'transcripts_dc.txt')
    change_word('cj', '씨제이', 'transcripts_dc.txt', 'transcripts_cj.txt')
    change_word('f', '에프', 'transcripts_cj.txt', 'transcripts_f.txt')
    change_word('ai', '에이아이', 'transcripts_f.txt', 'transcripts_ai.txt')
    change_word('a', '에이', 'transcripts_ai.txt', 'transcripts_a.txt')

    remove_english_word('transcripts_a.txt', 'transcripts_korean.txt')

if __name__ == '__main__':
    make_raw_transcript()
    modify_raw_transcript()
    
    # Vectorize
    vector = make_vector('transcripts_korean.txt')
    write_vector_file(vector, 'vector.txt')

    print('Finished')

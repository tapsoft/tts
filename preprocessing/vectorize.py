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

if __name__ == '__main__':
    vector = make_vector('transcripts_korean_final.txt')
    print(len(vector), vector)
    write_vector_file(vector, 'vector.txt')
    read_vector = read_vector_file('vector.txt')
    print(len(read_vector), read_vector)
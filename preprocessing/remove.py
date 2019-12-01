if __name__ == '__main__':
    script_filename = 'transcripts_a.txt'
    output_filename = 'transcripts_korean.txt'
    with open(script_filename, 'r') as f:
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
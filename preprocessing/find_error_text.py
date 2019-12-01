with open('transcripts_korean_final.txt', 'r') as f:
    lines = f.readlines()
    errors = ['3', '1', '4', '2', '-', '8', '7', '5', '%', '6', '#', '0']
    ctr = 0
    for line in lines:
        [filename, text] = line.split('|')
        text = text.rstrip()
        length = len(text)
        hasError = False
        for i in range(length):
            w = text[i]
            if w in errors:
                hasError = True
        if hasError:
            print(filename + "|" + text)
            ctr = ctr + 1
    print(ctr)
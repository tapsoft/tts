########### decomposition to syllables ###########
# to use, replace text_to_sequence(text) in __init.py__ to
'''
def text_to_sequence(text):
    r_lst, idx_list = phoneme(text)
    return [nsymbols] + idx_list + [nsymbols + 1]
import numpy as np
'''

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
CHOSUNG_IDX = list(np.arange(0, 19))

# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                 'ㅣ']
JUNGSUNG_IDX = list(np.arange(0, 21) + 19)

# 종성 리스트. 00 ~ 27 + 1 (1개 없음)
JONGSUNG_LIST = ['CODA', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JONGSUNG_IDX = list(np.arange(0, 29) + 19 + 21)

# 특수문자 리스트, 00 ~ 06 (공백 포함)
SPECIAL_LIST = ['_', ' ', ',', '.', '?', '!']
SPECIAL_IDX = list(np.arange(0, 7) + 19 + 21 + 29)

# UNKNOWN
UNKNOWN_IDX = 19 + 21 + 29 + 7

# Complete mapping
symbols = CHOSUNG_LIST + JUNGSUNG_LIST + JONGSUNG_LIST + SPECIAL_LIST + ["UNK"]
idx = CHOSUNG_IDX + JUNGSUNG_IDX + JONGSUNG_IDX + SPECIAL_IDX + [UNKNOWN_IDX]

'''
print(CHOSUNG_IDX)
print(JUNGSUNG_IDX)
print(JONGSUNG_IDX)
print(SPECIAL_IDX)
print(UNKNOWN_IDX)

print(symbols)
print(idx)
'''


def phoneme(sentence):
    r_lst = []
    idx_list = []

    for w in list(sentence.strip()):
        # 영어인 경우 구분해서 작성함.

        if '가' <= w <= '힣':
            # 588개 마다 초성이 바뀜.
            ch1 = (ord(w) - ord('가')) // 588
            # 중성은 총 28가지 종류
            ch2 = ((ord(w) - ord('가')) - (588 * ch1)) // 28
            ch3 = (ord(w) - ord('가')) - (588 * ch1) - 28 * ch2
            r_lst = r_lst + [CHOSUNG_LIST[ch1], JUNGSUNG_LIST[ch2], JONGSUNG_LIST[ch3]]
            idx_list = idx_list + [CHOSUNG_IDX[ch1], JUNGSUNG_IDX[ch2], JONGSUNG_IDX[ch3]]

        elif w in SPECIAL_LIST:
            r_lst.append(w)
            idx_list.append(SPECIAL_IDX[SPECIAL_LIST.index(w)])

        else:
            r_lst.append('UNK')
            idx_list.append(UNKNOWN_IDX)

    return r_lst, idx_list


'''
# Identify special characters in dataset
symbols = '/home/cs470/zeroshot-tts-korean/tacotron2/text/symbols.txt'
f = open(symbols, 'r')
symbols = list(f.readline().split('|'))
special = []
for s in symbols:
    if not ('가' <= s <= '힣'):
        special.append(s)
print(special)
'''

'''
filename = '/home/cs470/zeroshot-tts-korean/tacotron2/filelists/transcripts_korean_final_validate.txt'
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('|')[1]
        line = line.rstrip()
        symbols, idx = phoneme(line)
        print(line)
        print(symbols)
        print(idx)
'''

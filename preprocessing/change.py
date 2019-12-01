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


if __name__ == '__main__':
    # change_word('pc', '피씨', 'transcripts2.txt', 'transcripts_pc.txt')
    # change_word('tv', '티비', 'transcripts_pc.txt', 'transcripts_tv.txt')
    # change_word('b', '비', 'transcripts_tv.txt', 'transcripts_b.txt')
    # change_word('c', '씨', 'transcripts_b.txt', 'transcripts_c.txt')
    # change_word('lg', '엘지', 'transcripts_c.txt', 'transcripts_lg.txt')
    # change_word('sns', '에스엔에스', 'transcripts_lg.txt', 'transcripts_sns.txt')
    # change_word('cgv', '씨지비', 'transcripts_sns.txt', 'transcripts_cgv.txt')
    # change_word('pt', '피티', 'transcripts_cgv.txt', 'transcripts_pt.txt')
    # change_word('mt', '엠티', 'transcripts_pt.txt', 'transcripts_mt.txt')
    # change_word('ot', '오티', 'transcripts_mt.txt', 'transcripts_ot.txt')
    # change_word('asmr', '에이에스엠알', 'transcripts_ot.txt', 'transcripts_asmr.txt')
    # change_word('ktx', '케이티엑스', 'transcripts_asmr.txt', 'transcripts_ktx.txt')
    # change_word('d', '디', 'transcripts_ktx.txt', 'transcripts_d.txt')
    # change_word('kt', '케이티', 'transcripts_d.txt', 'transcripts_kt.txt')
    # change_word('sk', '에스케이', 'transcripts_kt.txt', 'transcripts_sk.txt')
    # change_word('dc', '디씨', 'transcripts_sk.txt', 'transcripts_dc.txt')
    # change_word('cj', '씨제이', 'transcripts_dc.txt', 'transcripts_cj.txt')
    # change_word('f', '에프', 'transcripts_cj.txt', 'transcripts_f.txt')
    # change_word('ai', '에이아이', 'transcripts_f.txt', 'transcripts_ai.txt')
    # change_word('a', '에이', 'transcripts_ai.txt', 'transcripts_a.txt')
    print('finished')
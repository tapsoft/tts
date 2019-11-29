with open('tacotron2/text/symbols.txt', 'r') as f:
  line = f.readline()
  line = line.rstrip()
  syms = line.split('|')

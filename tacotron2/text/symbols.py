with open('text/symbols.txt', 'r') as f:
  line = f.readline()
  line = line.rstrip()
  symbols = line.split('|')
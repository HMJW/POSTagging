fin = "../data/PTB/test.conllx"
fout = "./data/PTB/test.txt"

fin = open(fin, "r")
fout = open(fout, "w")
for line in fin:
    if line == "\n":
        fout.write("\n")
    else:
        line = line.strip().split()
        word, pos = line[1], line[3]
        fout.write(f"{word}\t{pos}\n")
fin.close()
fout.close()
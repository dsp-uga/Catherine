# opcode n-grams

# generate 3-grams dictionary for one file
def ngrams_dictionary(filename, N=3):
    op_list = []
    # p = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]+)')
    with open(filename, 'rb') as f:
        for line in f:
            op_list += str(line).rstrip().split(" ")[1:]
    # print(op_list)
    grams_string = [''.join(op_list[i:i+N]) for i in range(len(op_list)-N+1)]
    ngrams = dict()
    for gram in grams_string:
        if gram not in ngrams:
            ngrams[gram] = 1
    return ngrams

file = "files/01SuzwMJEIXsK7A8dQbl.asm"
# ngrams_dictionary(file)
print(ngrams_dictionary(file))

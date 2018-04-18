import bz2
import pickle


# file = "/Volumes/RedditData/train-unbalanced.csv.bz2"


# Read label, comment and author from file
def getData(file):
    with bz2.BZ2File(file, "r") as bzfin:
        for line in bzfin:
            label, comment, author = line.decode('utf-8').split('\t')[0:3]
            yield label, comment, author


# Aggregate user comments by creating dictionary:
# user: [(comment, label)]
def getUserDict(generator):
    from collections import defaultdict
    users = defaultdict(list)
    for i in list(generator):
        users[i[2]].append((i[1], i[0]))
    return users


# Write objects to pickle
def pickleFile(path, obj):
    file = open(path, 'w')
    pickle.dump(obj, file)
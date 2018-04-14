import bz2


file = "/Volumes/RedditData/train-unbalanced.csv.bz2"

# Read label, comment and author from file
def getData(file):
    with bz2.BZ2File(file, "r") as bzfin:
        lines = []
        for i, line in enumerate(bzfin):
            # Yield 2000 lines of data
            if i == 2000:
                break
            label, comment, author = line.decode('utf-8').split('\t')[0:3]
            yield label, comment, author




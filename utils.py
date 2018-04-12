import bz2
import re


file = "/Volumes/RedditData/train-unbalanced.csv.bz2"

with bz2.BZ2File(file, "r") as bzfin:
    lines = []
    for i, line in enumerate(bzfin):
        if i == 1:
            break
        print(line)









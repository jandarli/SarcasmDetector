import utils

if __name__ == '__main__':
    generator = utils.getData("/Volumes/RedditData/train-balanced.csv.bz2")
    users = utils.getUserDict(generator)
    utils.pickleFile("data.txt", users)

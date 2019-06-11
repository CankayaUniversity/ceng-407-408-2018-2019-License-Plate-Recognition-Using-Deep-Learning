import os
import glob


names = [os.path.basename(x) for x in glob.glob('/home/utku/Desktop/Mert/bitirme/*')]

names.sort()
F = open("labels.txt","w") 



def cmp_func(a, b):
    # sort by length and then alphabetically in lowercase
    if len(a) == len(b):
        return cmp(a, b)
    return cmp(len(a), len(b))

sorted_the_way_you_want = sorted(names, cmp=cmp_func)


print sorted_the_way_you_want

for filename in sorted_the_way_you_want:
	F.write(filename[:-4]+ '\n')


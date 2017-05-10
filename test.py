# Dividing dataset
import string
f = open("Friends-dialogues.txt",'r')
f2 = open("F1.txt",'w')
i = 0

replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))

for l in f:
	if i>10000:
		break
	f2.write(l.translate(replace_punctuation))
	i+=1
f.close()
f2.close()
# f3 = open("hi.txt",'w')
# for i in range(10000):
# 	f3.write("hi there\nyes\nhere\nno\n")
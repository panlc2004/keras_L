import csv
import numpy as np

filename = 'train_data.csv'
with open(filename) as f:
    reader = csv.reader(f)
    list = list(reader)[0:1]
for i in list:
    s = i[0]
    s = s.replace("b'", "").replace("[", "").replace("]", "")
    s = s.split(" ")
    print(s[0])
    print(s[1])
    print(i[1])



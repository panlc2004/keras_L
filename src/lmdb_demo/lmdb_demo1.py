import lmdb
import json
import time
import numpy as np

env = lmdb.open("database")
# # 注意用txn = env.begin()创建事务时，有write = True才能够写数据库。
a = time.time()

for i in range(100):
    txn = env.begin(write=True)

    txn.put('test'.encode(), 'ttt'.encode())
    txn.commit()

b = time.time()

print('time:', b - a)

# a = [1,2,3]
# b = np.asarray(a)
# c = b.dumps()
# for i in range(800):
#     d = np.loads(c)

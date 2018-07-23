import sqlite3
import numpy as np
import time

import keras.preprocessing.image as image

conn = sqlite3.connect('test.db')
path = 'E:\GitCode\python\keras_L\src\image_load\\001.jpg'
a = image.load_img(path, target_size=[96, 96])
a = image.img_to_array(a) / 255.
a = a.dumps()

def create_table():
    sql = """
    CREATE TABLE "train" (
    "path"  TEXT NOT NULL,
    "data"  BLOB,
    PRIMARY KEY ("path")
    );

    CREATE UNIQUE INDEX "un_path"
    ON "train" ("path" ASC);
    """
    cur = conn.cursor()
    cur.execute(sql)

def insert(path, data, conn):
    sql = 'INSERT INTO train VALUES(?,?)'
    cur = conn.cursor()
    cur.execute(sql, (path, data))


def find(path, conn):
    sql = "select *  from train where path='%s'" % path
    cur = conn.cursor()
    cur.execute(sql)
    values = cur.fetchall()
    return values

def delete(conn):
    sql ="delete from train"
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()

def find2(path, conn):
    sql = "select *  from train where path=?"
    cur = conn.cursor()
    cur.execute(sql, path)
    values = cur.fetchall()
    return values

# delete(conn)

# for i in range(500000):
#     print(i)
#     insert(path + str(i), a, conn)
#     if i % 10000 == 0 and i != 0:
#         conn.commit()


x = np.random.randint(0,20000,1000)
# y = [path + str(i) for i in x]
a = time.time()
for i in x:
    s = find(path + str(i), conn)
    d = s[0][1]
    d = np.loads(d)
    print(d.shape)
b = time.time()
print(b - a)

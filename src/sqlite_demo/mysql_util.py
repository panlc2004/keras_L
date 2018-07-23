import pymysql.cursors
import keras.preprocessing.image as image


class MysqlUtil:
    def __init__(self):
        self.__connect()

    def __connect(self):
        self.conn = pymysql.connect(host='10.131.100.6',
                                    port=3306,
                                    user='root',
                                    password='000000',
                                    db='casia_webface',
                                    charset='utf8',
                                    cursorclass=pymysql.cursors.DictCursor)
        self.cursor = self.conn.cursor(cursor=pymysql.cursors.DictCursor)

    def insert(self, table, path, data):
        sql = 'INSERT INTO %s VALUES(%s,%s)' % (table,'%s','%s')
        self.cursor.execute(sql, (path, data))

    def find_one(self, table, path):
        sql = "select `data` from %s where path='%s'" % (table, path)
        self.cursor.execute(sql)
        values = self.cursor.fetchone()
        return values

    def find_list(self, table, path):
        sql = "select path,`data` from %s where path in " % table
        param = self.__build_in_param(path)
        sql = sql + param
        self.cursor.execute(sql)
        values = self.cursor.fetchall()
        return values

    def __build_in_param(self, list_param):
        p = '"1"'
        for param in list_param:
            param = '"' + param + '"'
            p = p + ',' + param
        res = '(' + p + ')'
        return res

    def total_num(self, table):
        sql = "select COUNT(1) from %s" % table
        self.cursor.execute(sql)
        values = self.cursor.fetchone()
        return values

    def insert_if_not_exist(self, table, path, data):
        if self.find_one(table, path) is None:
            self.insert(table, path, data)

    def commit(self):
        try:
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    def close(self):
        self.cursor.close()
        self.conn.close()


util = MysqlUtil()
path = 'D:\\panlc\\Data\\CASIA-WebFace\\0000045\\001.jpg'
# a = image.load_img(path, target_size=[96, 96])
# a = image.img_to_array(a) / 255.
# a = a.dumps()
#
# util.insert('train', path, a)
# util.commit()

paths = [path, 'b', 'c']
# s = util.find_list('train', paths)
s = util.find_one('train', path)
print(s)

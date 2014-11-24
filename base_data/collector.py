import sqlite3
import json
import re
import string


class Collector(object):
    def __init__(self):
        self.db = sqlite3.connect('generic_tweets.db')
        self.cursor = self.db.cursor()
        self.cursor.executescript("""DROP TABLE IF EXISTS generic_tweets;
            CREATE TABLE generic_tweets(t TEXT);""")

    def __insert_to_db(self, text):
        self.cursor.execute("INSERT INTO generic_tweets (t) VALUES(?)",
                    (text.decode('utf-8'),))

    def get_from_file(self, filename):
        with open(filename) as f:
            for line in f:
                print line
                if line != '':
                    try:
                        tweet = json.loads(line)
                    except ValueError:
                        continue
                    else:
                        if 'text' in tweet:
                            self.__insert_to_db(tweet['text'].encode('utf-8'))
        print "Commiting!"
        self.db.commit()

    def close(self):
        self.cursor.close()
        self.db.close()


# c = Collector()
# c.get_from_file('base_data.txt')
# c.close()

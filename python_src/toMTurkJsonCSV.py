'''
Writes tab-separated tweet ID and tweet text pairs to single-column CSV.
Value in the CSV is a JSON object with 10 tweets.

'''

import codecs, cStringIO, csv, json, sys

class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

#csv.register_dialect('mturk', delimiter=',', quoting=csv.QUOTE_MINIMAL,
#                     doublequote=False, escapechar='\\',
#                     lineterminator='\n')

inPath = sys.argv[1]
outPath = sys.argv[2]

lineIdx = 0

outFile = open(outPath, 'w')
#writer = csv.writer(outFile, dialect='mturk')
writer = UnicodeWriter(outFile, dialect='excel')
writer.writerow(['tweets']) # Header line

currJsonObj = {}
currIdx     = 1
f = codecs.open(inPath, 'r', encoding='utf8')
for lineIdx, line in enumerate(f):
  if not lineIdx % 10000:
    print 'Line %d*10K' % (lineIdx/10000)
  
  flds = line.strip().split('\t')
  if len(flds) < 2:
    continue
  
  tid, text = int(flds[0]), flds[1]
  currJsonObj['tweet_id_%d' % (currIdx)] = tid
  currJsonObj['tweet_text_%d' % (currIdx)] = text
  
  if not (currIdx % 10):
    tweets = json.dumps(json.dumps(currJsonObj))
    writer.writerow([tweets])
    
    currJsonObj = {}
    currIdx = 0
  
  currIdx += 1
f.close()
outFile.close()

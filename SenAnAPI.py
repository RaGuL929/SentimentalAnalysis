import tracemalloc
import linecache
import os
from flask import request,Flask
import SentimentalAnalysingAPI
from flask_restful import Resource, Api
def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

app = Flask(__name__)
app.config["DEBUG"] = True
api = Api(app)


class get_sentiment(Resource):
    def get(self):
        if 'text' in request.args:
            text=request.args['text']
            # tracemalloc.start()
            list1=list(SentimentalAnalysingAPI.predict_text(text))

            # display_top(tracemalloc.take_snapshot())
            return{'Sentiment':list1[0],'Confidence':list1[1]}
        else:
            return{'Output':'No Input'}



api.add_resource(get_sentiment, '/get_sentiment/')

if __name__ == '__main__':
     app.run()


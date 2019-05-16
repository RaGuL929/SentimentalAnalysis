from flask import request,Flask
import SentimentalAnalysingAPI
from flask_restful import Resource, Api

 
app = Flask(__name__)
app.config["DEBUG"] = True
api = Api(app)


class get_sentiment(Resource):
    def get(self):
        if 'text' in request.args:
            text=request.args['text']
            list1=list(SentimentalAnalysingAPI.predict_text(text))
            return{'Sentiment':list1[0],'Confidence':list1[1]}
        else:
            return{'Output':'No Input'}



api.add_resource(get_sentiment, '/get_sentiment/')

if __name__ == '__main__':
     app.run()


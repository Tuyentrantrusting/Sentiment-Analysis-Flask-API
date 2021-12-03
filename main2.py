
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
api = Api(app)


output = {}

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('q', help='Pass a sentence to analyse')


class SentimentAnalysis(Resource):
    ''' 
        Analyse the text for positive or negative sentiment
        API for Google Cloud Deployment
    '''

    def get(self):

        # use parser to find the user's query
        args = parser.parse_args()
        sentence = args['q']

        # analyse supplied query
        nltk.download("vader_lexicon")
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(sentence)["compound"]
        if score > 0:
            output['sentiment'] = 'positive'
        else:
            output['sentiment'] = 'negative'

        return jsonify(output)

# use for flask_restful
api.add_resource(SentimentAnalysis, '/')

if __name__ == "__main__":
    app.run()


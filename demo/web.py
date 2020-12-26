#!C:\Users\tarel\Anaconda3\python.exe

from classifier import Classifier, text_preprocessor
from flask import Flask, render_template, request
import sys

app = Flask(__name__)

print('Load classifier', file=sys.stderr)
classifier = Classifier()
print('Classifier is successfully loaded', file=sys.stderr)


@app.route('/dialogues-demo', methods=['POST', 'GET'])
def index_page(text='', prediction_message=''):
    if request.method == 'POST':
        text = request.form['text']

        prediction_message = classifier.get_result_message(text)

    return render_template('simple_page.html', text=text, prediction_message=prediction_message)


if __name__ == '__main__':
    app.run()

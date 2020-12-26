import pickle
import pandas as pd
import re
import string

from nltk.stem import WordNetLemmatizer

MODEL_FILE_PATH = 'model_dump.pkl'
VECTORIZER_FILE_PATH = 'dialogs_vectorizer.pkl'
THRESHOLD_FILE_PATH = 'threshold.csv'


def text_preprocessor(text):
    text = re.sub(f'[{string.punctuation}(<BR>)]', '', text)
    text = text.lower()
    lemmatizer = WordNetLemmatizer()
    lem_text = [lemmatizer.lemmatize(x) for x in re.findall(r'\b\w\w+\b', text)]

    return ' '.join(lem_text)


class Classifier:
    def __init__(self):
        self.vectorizer = self.load_pickle(VECTORIZER_FILE_PATH)
        self.model = self.load_pickle(MODEL_FILE_PATH)
        self.df_threshold = pd.read_csv(THRESHOLD_FILE_PATH, index_col=0)
        self.target_names = self.df_threshold.index

    @staticmethod
    def load_pickle(file_path):
        with open(file_path, 'rb') as f_in:
            result = pickle.load(f_in)

        return result

    def get_name_by_label(self, label):
        try:
            return self.target_names[label]
        except IndexError:
            return "label error"

    def predict_dialogue(self, dialogue):
        vectorized = self.vectorizer.transform([dialogue])
        preds = self.model.predict(vectorized)[0]

        for n, yi in enumerate(self.target_names):
            preds[n] = preds[n] > self.df_threshold.loc[yi, 'best_threshold']

        return preds

    def get_result_message(self, dialogue):
        prediction = self.predict_dialogue(dialogue)
        result = [self.get_name_by_label(i) for i, n in enumerate(prediction) if n]
        return ' '.join(result)


if __name__ == '__main__':
    with open('dialogue_example_good_will_hunting.txt', 'r') as f_in:
        dialogue_example = f_in.read()

    classifier = Classifier()
    print(Classifier().get_result_message(dialogue_example))

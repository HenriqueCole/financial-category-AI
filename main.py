from cProfile import label
from pyexpat import model
from gravityai import gravityai as ga
import pickle
import pandas as pd

model = pickle.load(open('financial_text_classifier.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('financial_text_vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('financial_text_encoder.pkl', 'rb'))


def process(inPath, outPath):
    input_df = pd.read_csv(inPath)
    features = tfidf_vectorizer.transform(input_df['body'])
    predictions = model.predict(features)
    input_df['caterogy'] = label_encoder.inverse_transform(predictions)
    output_df = input_df[['id', 'category']]
    output_df.to_csv(outPath, index=False)


grav.wait_for_requests(process)



import re
import sys
import pandas as pd
import time
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from torchtext.data.utils import get_tokenizer
from nltk.util import ngrams
import nltk
from collections import Counter
from bs4 import BeautifulSoup
import html
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
import networkx as nx
import os
import matplotlib.pyplot as plt
from global_custom_functions import clean_and_split_text
import classifier_7 as cl
import datetime


sys.path.append('D:/info/uni_tubingen/02 Master/4. Semester/Thesis/Thesis/scripts/data_processing/sentence classifier')
sys.path.append('D:/info/uni_tubingen/02 Master/4. Semester/Thesis/Thesis/scripts/')


os.getcwd()
os.chdir('D:/info/uni_tubingen/02 Master/4. Semester/Thesis/Thesis/scripts/')
print("CWD is: ", os.getcwd())


# Create NN model for classification

model = cl.Model()
model.create_model()


class Job_text():
    def __init__(self, text):
        self.text = text
        self.sentences = None
        self.labels = []
        self.get_sentences()
        self.get_labels()

    def get_sentences(self):
        free_text = self.text
        self.sentences = clean_and_split_text(free_text)

    def get_labels(self):
        self.labels = []
        for each in self.sentences:
            self.labels.append(model.predict(model.vectorizer.transform([each]).toarray()))

    def return_category(self, category):
        output = []
        for i, each in enumerate(self.labels):
            if each == category:
                output.append(self.sentences[i])
        return output


jobs = pd.read_csv('./data_preprocessing/preprocessed_jobs.csv')

print("Nr. of raw jobs loaded: ", jobs.shape[0])

jobs_to_load = jobs.shape[0]
skillset_list_raw = {}



# jobs_to_load = 100

def update_jobs(i, skillset_list_raw):
    job_i = Job_text(jobs.iloc[i, :].content)

    job_i.labels
    job_i.get_labels()


    skillset_list_raw[i] = {}
    skillset_list_raw[i]['title'] = (jobs.iloc[i].jobTitle)
    skillset_list_raw[i]['text'] = []
    # job_i.return_category(2)
    # [skillset_list_raw[i]['text'].append(each) for each in job_i.return_category(3)]
    [skillset_list_raw[i]['text'].append(each) for each in job_i.return_category(2)]

    return skillset_list_raw

for item in range(jobs_to_load):
    skillset_list_raw = update_jobs(item, skillset_list_raw)
    if item % 100 == 0:
        print(item, '/', str(jobs_to_load))


list(skillset_list_raw)



tick1 = datetime.datetime.now()

the_range = np.random.choice(jobs.shape[0], 1000, replace=False)
the_range = np.arange(jobs.shape[0])


df1 = pd.DataFrame(columns={"title", "text"})
job_i = jobs.iloc[the_range, :].content
job_i_text = job_i.apply(lambda x: Job_text(x).return_category(2))


tick2 = datetime.datetime.now()
print(tick2 - tick1)



df1.title = jobs.iloc[the_range, :].jobTitle
df1.text = job_i_text

skillset_list_clean = []
for i in df1['text']:
    all_sentences = ''
    for sent in i:
        sent = sent.lower()
        sent = BeautifulSoup(html.unescape(sent), 'lxml').text
        sent = sent.replace('\n', ' ')
        sent = sent.replace('?', ' ')
        sent = sent.replace('!', ' ')
        sent = sent.replace('.', ' ')
        sent = sent.replace('&', ' ')
        sent = sent.replace(',', ' ')
        sent = sent.replace(';', ' ')
        sent = sent.replace('(', ' ')
        sent = sent.replace(')', ' ')
        sent = sent.replace('·', ' ')
        sent = sent.replace('•', ' ')
        sent = sent.replace(';', ' ')
        sent = sent.replace('•', ' ')
        sent = sent.replace('*', ' ')
        sent = sent.replace('-', '')
        sent = sent.replace('/', ' ')
        sent = sent.replace('\\', '')
        sent = sent.split()

        sent_clean = sent
        sent_trans = str(sent_clean).strip("'[]'")
        sent_trans = sent_trans.replace("'", "")
        sent_clean = sent_trans.replace(",", "")

        all_sentences = all_sentences + ' ' + sent_clean
    skillset_list_clean.append(all_sentences)

print('Size of skillset list: ', len(skillset_list_clean))

df1['text_clean'] = skillset_list_clean





df1.to_csv('./data_processing/topic modelling/data/df_skillset_clean4.csv')







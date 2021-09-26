
import os, sys
sys.path.append('C:/Users/Maria/PycharmProjects/Thesis/scripts/')

from global_custom_functions import clean_and_split_text
import pandas as pd
import re
from datetime import datetime
import numpy as np

while True:

    jobs = pd.read_csv('./scripts/data_preprocessing/preprocessed_jobs.csv')
    categories = {1:'tasks', 2:'skillset', 3:'company', 4:'other'}
    dic_labeled_sentences = {'sentence':[],'jobTitle':[],'company':[],'estimatedDatePosted':[], 'label':[]}

    unique_jobs = jobs.drop_duplicates(['jobTitle','company', 'estimatedDatePosted' ])

    random_sampe = np.random.choice(unique_jobs.shape[0],1)

    for random_draw in random_sampe:
        free_text = unique_jobs.iloc[random_draw]['content']
        sentences = clean_and_split_text(free_text)
        print('_______________________ neue Stelle _________________\n')
        print(unique_jobs.iloc[random_draw]['jobTitle'], ', len sent:',len(sentences))
        print('\n')
        to_continue = 'y'
        to_continue = input('>> Press "n" to go to next job: ')
        if to_continue == 'n':
            continue
        for i in range(len(sentences)):
            each = sentences[i]
            if len(each) < 3:
                continue
            print('_____________________________________\n')
            print(each)
            label = ""
            while len(re.findall(r'\d', label))!=1:
                label = str(input(f">>> Which category is this sentence? Enter the number: {categories} "))
            label_sent = int(label)
            dic_labeled_sentences['sentence'].append(each)
            dic_labeled_sentences['label'].append(categories[label_sent])
            dic_labeled_sentences['jobTitle'].append(unique_jobs.iloc[random_draw]['jobTitle'])
            dic_labeled_sentences['estimatedDatePosted'].append(unique_jobs.iloc[random_draw]['estimatedDatePosted'])
            dic_labeled_sentences['company'].append(unique_jobs.iloc[random_draw]['company'])

    df_labels = pd.DataFrame().from_dict(dic_labeled_sentences, orient='columns')

    str_time = str(datetime.now().month) + "-" + str( datetime.now().day) + "-" + str(datetime.now().minute) + "-" + str(datetime.now().second)
    df_labels.to_csv(f'./scripts/data_preprocessing/training_datasets/labeled_data_{str_time}.csv', encoding='utf-8')


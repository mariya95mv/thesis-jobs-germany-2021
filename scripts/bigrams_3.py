globals().clear()

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
import matplotlib.pyplot as plt
import os

sys.path.append('D:/info/uni_tubingen/02 Master/4. Semester/Thesis/Thesis/scripts/data_processing/sentence classifier')
sys.path.append('D:/info/uni_tubingen/02 Master/4. Semester/Thesis/Thesis/scripts/')
os.chdir('D:/info/uni_tubingen/02 Master/4. Semester/Thesis/Thesis/scripts/')

from global_custom_functions import clean_and_split_text


os.getcwd()
# Load NN model and skillset data here

df_skillset = pd.read_csv('./data_processing/topic modelling/data/df_skillset_clean4.csv')
df_skillset.columns
df_skillset = df_skillset.drop(["Unnamed: 0"], axis=1)


def clean_titles(each):
    each = each.replace('mit', '').lower()
    each = each.replace('in', '').lower()
    each = each.replace('im', '').lower()
    each = each.replace('by', '').lower()
    each = each.replace('zum', '').lower()
    each = each.replace('of', '').lower()
    each = each.replace('and', '').lower()
    each = each.replace('für', '').lower()
    each = each.replace('und', '').lower()
    each = each.replace('neu', '')
    each = each.replace('ey', '')
    each = each.replace('all genders', '')
    each = str(re.findall(r'\w{2,}', each)).replace("'", '').replace("[", '').replace("]", '').replace(",", '')
    return each

    # texts_clean.append(each)


def detect_language(text):
    text_tok = text.split()
    german_words = ['für', "vom", "von", "mit", "zeit", "erfahr", "deine", "dein", "sie","bis","zu", "uber"]
    for word in german_words:
        if word in text_tok:
            return "GER"
    return "ENG"


len(df_skillset['text_clean'])
to_drop = df_skillset['text_clean'].apply(lambda x: pd.isna(x))
df_skillset = df_skillset[~to_drop].reset_index()

df_skillset['lang'] = df_skillset["text_clean"]
df_skillset['lang'] = df_skillset['lang'].apply(lambda x: detect_language(x))

print("German texts in data: ")
print((df_skillset['lang'] == "GER").sum() / df_skillset.shape[0])

df_skillset = df_skillset[df_skillset['lang'] == "GER"]

df_skillset['text_clean'] = ' SOS ' + df_skillset['text_clean'] + ' EOS'
df_skillset['text_clean'] = df_skillset['text_clean'].apply(lambda x: clean_titles(x))

from nltk.corpus import stopwords

german_stop_words = stopwords.words('german')
german_stop_words.extend(['dee', 'ee', 'von', 'oder', 'ees', 'sie', 'der', 'de','sd',
                          'eem', 'eer', 'een', 'du', 'sich', 'gern', 'ihr', 'hast', 'sich', 'hoch', 'sowie', 'gut',
                          'gute', 'guter','gutes',
                          'idealerweise', 'mass', 'bringen', 'bringst', 'profil', 'erste', 'möchtest', "mocht",
                          "prildu", "profildu", "punktest", "punkten",
                          'schwerpunkt', 'innerhalb', 'mass', 'bereich',
                          'freust', 'freuen', 'brgen', 'hoher'])

text_field = 'du aufgaben du musst du bearteiten tabellen erstellen'


def remove_stopwords(text_field):
    sent = text_field.split()
    for word in german_stop_words:
        while word in sent:
            sent.remove(word)
    # print(sent)
    return str(sent).replace("[", "").replace("]", "").replace("'", "").replace(",", "")


text_field = list(df_skillset['text_clean'])[-1]
df_skillset['text_clean_short'] = df_skillset['text_clean'].copy()
df_skillset['text_clean_short'] = df_skillset['text_clean_short'].apply(lambda x: remove_stopwords(x))

to_drop = df_skillset['text_clean_short'].apply(lambda x: len(x.split()) < 5)

print("Removing descriptions with <5 words: ", np.sum(to_drop))

df_skillset = df_skillset[~to_drop].reset_index()
df_skillset = df_skillset.drop(['level_0', 'index'], axis=1).reset_index().drop('index', axis=1)


print('Dropping duplicates (title + text)')
df_skillset = df_skillset.drop_duplicates(['text', 'title']).reset_index().drop(['index'], axis=1)
print('_'*50,'\nFinished text preparation:')
print('-'*20,'SUMMARY STATS', '-'*20)
print("Average nr of words in document: ")
print(df_skillset['text_clean_short'].apply(lambda x: len(x.split())).mean())
print("Quantile-25 nr of words in document: ")
print(df_skillset['text_clean_short'].apply(lambda x: len(x.split())).quantile(0.25))
print("Quantile-75 nr of words in document: ")
print(df_skillset['text_clean_short'].apply(lambda x: len(x.split())).quantile(0.75))
print("Quantile-95 nr of words in document: ")
print(df_skillset['text_clean_short'].apply(lambda x: len(x.split())).quantile(0.95))




# get seniority levels -----------------------------------------------------------


column_titles = df_skillset.title
df_skillset['seniority'] = 'Unknown'
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('praktik..', x.lower())), "PR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('intern ', x.lower())), "PR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('aushilfe', x.lower())), "PR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('hilfskraft', x.lower())), "PR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('senior..', x.lower())), "SR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('sr', x.lower())), "SR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('junior..', x.lower())), "JR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('werkstudent..', x.lower())), "PR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('trainee', x.lower())), "JR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('hochschulabsolvent', x.lower())), "JR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('lead', x.lower())), "SR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('abteilungsleiter', x.lower())), "SR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('leiter', x.lower())), "SR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('team manager', x.lower())), "SR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('expert.', x.lower())), "SR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('director', x.lower())), "SR",
                                    df_skillset['seniority'])
df_skillset['seniority'] = np.where(column_titles.apply(lambda x: re.findall('teamleit.', x.lower())), "SR",
                                    df_skillset['seniority'])

print('-'*20, "Seniority stats:", '-'*40,'\n')
print(df_skillset.groupby(['seniority']).count())
print("df shape: ", df_skillset.shape[0])


#######################################
# create segments based on titles


##################################################3

# simple generative model: greedy algorithm, take most likely word. no smapling
def generate_sentence(df_text_trigrams):
    most_likely_sentence = []
    sentence_prob = []
    most_likely_sentence.extend(['sos', 'sos'])
    token_t1 = most_likely_sentence[-1]
    while token_t1 != 'eos':
        token_t1 = most_likely_sentence[-1]
        token_t0 = most_likely_sentence[-2]
        df_text_trigrams[df_text_trigrams['word_2'] == token_t1]
        wd_argmax = df_text_trigrams[(df_text_trigrams['word_1'] == token_t0) &
                                     (df_text_trigrams['word_2'] == token_t1)]
        if len(wd_argmax) < 1:
            print("Can't finish sentence. No words to continue with.")
            break
        idx_argmax = wd_argmax[wd_argmax['cond_prob'] == wd_argmax['cond_prob'].max()].index[0]
        next_token = df_text_trigrams['word_3'][idx_argmax]
        most_likely_sentence.append(next_token)
        sentence_prob.append(df_text_trigrams['cond_prob'][idx_argmax])
        # print(next_token)
        if len(most_likely_sentence) > 300:
            break
    print('sentence probability: ', np.product(sentence_prob))

    # df_text_bigrams[df_text_bigrams['word_2'] == 'team']
    return most_likely_sentence


#################################
# bigram model
def get_topics_bigram(df_skillset, field_name, seniority='Unknown'):
    column_to_filter = 'text_clean_short'

    if 'index' in df_skillset.columns:
        df_skillset = df_skillset.drop(['index'], axis=1)

    local_df = df_skillset.copy()
    marketing_jobs = np.where(local_df['title'].apply(lambda x: re.findall(field_name, x.lower())),
                              local_df.index,
                              np.nan)

    local_df['title'].apply(lambda x: re.findall(field_name, x.lower()))

    ma_index = marketing_jobs[pd.notna(marketing_jobs)]

    ma_index = ma_index.astype(int)
    df_subset = local_df.iloc[ma_index, :]

    texts = df_subset[column_to_filter]
    print("Nr. of jobs: ", texts.__len__())

    df_subset['title_clean'] = df_subset['title'].apply(lambda x: clean_titles(x))

    print(df_subset.groupby(['seniority']).count())
    texts = list(df_subset[local_df['seniority'] == seniority][column_to_filter])
    # texts.__len__()

    bi_counter = Counter()
    list_entries = [[]]

    local_df['text_clean_bigrams'] = np.nan

    for line_idx, line in enumerate(texts):
        list_entries.append([])
        token = nltk.word_tokenize(line)
        bigram = list(ngrams(token, 2))
        bigrams_ordered = []
        for bi_pair in bigram:
            # print(bi_pair)
            bi_pair_ordered = bi_pair[1] + '_' + bi_pair[0]
            bigrams_ordered.append(bi_pair_ordered)
            # print(bi_pair_ordered)
        local_df['text_clean_bigrams'].iloc[line_idx] = list(bigrams_ordered)
        for pair in bigram:
            bi_string = str(pair[1] + '_' + pair[0])
            bi_counter.update([bi_string])
            list_entries[line_idx].append(bi_string)

    bi_counter.items()
    bi_counter.__len__()
    bi_counter.most_common(10000)[-1]

    #####################
    df_text_trigrams = pd.DataFrame(columns=['word_1', 'word_2', 'count', 'word_1_abs_freq'])
    stem_germ = SnowballStemmer('german')

    token_counter = Counter()

    for i, item in enumerate(bi_counter.items()):
        if i % 2000 == 0:
            print(np.round(i / bi_counter.items().__len__(), 2))
        # print(i, '/', bi_counter.__len__())
        # print(item)
        words = re.findall(r'[^_]*', item[0])
        words_clean = []
        for w in words:
            if len(w) > 1:
                words_clean.append(stem_germ.stem(w))
        if len(words_clean) == 2:
            word_1 = words_clean[1]
            word_2 = words_clean[0]
            word_count = item[1]
            token_counter.update([word_1])

            if df_text_trigrams[(df_text_trigrams['word_1'] == word_1) & (df_text_trigrams['word_2'] == word_2)].shape[
                0] > 0:
                current_idx = \
                    df_text_trigrams[
                        (df_text_trigrams['word_1'] == word_1) & (df_text_trigrams['word_2'] == word_2)].index[
                        0]
                df_text_trigrams.iloc[current_idx, 2] += word_count
            else:
                df_text_trigrams = df_text_trigrams.append({'word_1': word_1,
                                                            'word_2': word_2,
                                                            'count': word_count}, ignore_index=True)

    # df_text_trigrams[(df_text_trigrams['word_2'] == 'digital') & (df_text_trigrams['word_1']=='marketg')]
    # df_text_trigrams[(df_text_trigrams['word_2'] == 'digital')].sort_values('count', ascending=0)

    df_text_trigrams['cond_prob'] = np.nan
    for wd in df_text_trigrams['word_1'].unique():
        wd1_array = (df_text_trigrams['word_1'] == wd)
        total_counts = df_text_trigrams['count'][wd1_array].sum()
        df_text_trigrams['cond_prob'][wd1_array] = df_text_trigrams['count'][wd1_array] / total_counts

    df_text_trigrams[df_text_trigrams['word_1'] == 'englisch']

    for i in range(df_text_trigrams['word_1'].shape[0]):
        abs_occurance = token_counter[df_text_trigrams['word_1'][i]]
        df_text_trigrams['word_1_abs_freq'][i] = abs_occurance

    return df_text_trigrams, token_counter, df_subset


###
# bigram


seniority = "Unknown"
field_name = 'market*'

df1 = df_skillset.copy()
# df1 = df1.iloc[:5000, ]

# define field names

field_names_finance = '(finan*)|(asset*)|(asset* manage*)|(cash)|(stock exchange)|(versicherung*)|(bank)|(wertpapier*)|(portfoliomanag*)|(risikomanag*)|(revenue)|(credit)|(risk)'
field_names_mkt = '(marketing)|(brand manager)|(commerce)|(social media)|(content)|(blog)|(influencer)|(crm)|(sea)|(seo)|(sem)|(advertising)|(customer success)'
field_names_accounting = '(audit*)|(accountant)|(accounting)|(rechnugswesen)|(tax)|(steuer*)|(controller)|(buchhalt*)|(bilanz*)|(gehaltsbuchhalter*)'
field_names_ds = '(data science)|(data engineer)|(data)|(analyst*)|(business intelligence)|(data scientist)'
field_names_consulting = '(berat*)|(consult*)'
field_names_sales = '(vertrieb*)|(sales )|(kundenberat*)|(kundenbetreu*)'
field_names_supply = '(supply chain)|(operations)|(supply chain management)'
field_names_pr = '(PR)|(relations)|(kommunikation )|(communication)|(unternehmenskommunikation)'
field_names_hr = '(HR)|(recruiter)|(human ressource)|(key account manag*)|(personalrekrutierung)|(personalentwickler*)|(recruiting)|(personalwesen)|(personaldisponent)'
field_names_projmag = '(project manage*)|(projektmanag*)|(agil*)|(projektleiter*)|(produktmanager*)|(product manager)|(projektkoordinator*)|(Prozessmanag*)'

########################################################
########################################################
########################################################

# Here we create datasets for each of the fields above

print(field_names_finance)
fin_prob_bigram, fin_token_count, df_fin = get_topics_bigram(df1, field_names_finance, seniority)


print(field_names_mkt)
mkt_prob_bigram, mkt_token_count, df_mkt = get_topics_bigram(df1,
                                                             field_names_mkt,
                                                             seniority)
print(field_names_accounting)
acc_prob_bigram, acc_token_count, df_acc = get_topics_bigram(df1, field_names_accounting,
                                                             seniority)

print(field_names_ds)
dat_prob_bigram, dat_token_count, df_dat = get_topics_bigram(df1,
                                                             field_names_ds,
                                                             seniority)
print(field_names_consulting)
cons_prob_bigram, cons_token_count, df_cons = get_topics_bigram(df1, field_names_consulting, seniority)

print(field_names_sales)
sale_prob_bigram, sale_token_count, df_sale = get_topics_bigram(df1, field_names_sales, seniority)

print(field_names_supply)
sup_prob_bigram, sup_token_count, df_sup = get_topics_bigram(df1, field_names_supply,
                                                             seniority)

print(field_names_pr)
pr_prob_bigram, pr_token_count, df_pr = get_topics_bigram(df1, field_names_pr, seniority)

print(field_names_hr)
hr_prob_bigram, hr_token_count, df_hr = get_topics_bigram(df1,
                                                          field_names_hr,
                                                          seniority)

print(field_names_projmag)
proj_prob_bigram, proj_token_count, df_proj = get_topics_bigram(df1, field_names_projmag,
                                                                seniority)


# execute until here




####################################################################################33
####################################################################################33
####################################################################################33

# what are the most frequent words connected to?

# keep only top n most common words in all documents (occur in the most documents)
def keep_top_n(dataframe, n=10):
    dummy_count = 0
    word = ''
    dataframe = dataframe.sort_values(['word_1_abs_freq', 'word_1'], ascending=False).reset_index().drop('index',
                                                                                                         axis=1)
    for i in range(dataframe.shape[0]):
        if dataframe['word_1'][i] != word:
            word = dataframe['word_1'][i]
            dummy_count += 1
            if dummy_count > n + 1:
                subset = dataframe.iloc[:i, :]
                return subset



list_bi_prob = [mkt_prob_bigram, fin_prob_bigram, cons_prob_bigram, sale_prob_bigram, hr_prob_bigram, proj_prob_bigram,
                sup_prob_bigram, dat_prob_bigram, acc_prob_bigram,
                pr_prob_bigram]

list_fields_names = ['marketing', 'finance','consulting','sales','hr','proj','supply_chain','data','acounting',
                     'pr']

for field_i, dataset in enumerate(list_bi_prob):

    print(list_fields_names[field_i])

    dataset = dataset.sort_values(['word_1_abs_freq', 'count'], ascending=False).reset_index().drop('index',
                                                                                                                    axis=1)
    fin_filter = keep_top_n(dataset, n=20)
    len(fin_filter['word_1'].unique())
    fin_filter = fin_filter[fin_filter['word_1'] != 'sos']

    # remove combinations that appear in less than 10 documents
    fin_filter = fin_filter[fin_filter['count'] > 10]
    fin_filter = fin_filter.reset_index().drop('index', axis=1)
    top100 = fin_filter

    dic_top100 = {}
    for i in range(top100.shape[0]):
        dic_top100[tuple(top100.iloc[i, :2])] = top100.iloc[i, 2]

    ###################
    # plot word frequency
    x_axis = fin_filter['word_1']
    y_axis = fin_filter['word_1_abs_freq']

    plt.figure(figsize=(12, 8))
    plt.bar(x_axis, y_axis)
    plt.title(f"Word frequencies for {list_fields_names[field_i]}")
    plt.xticks(rotation=40)

    plt.show()
    plt.savefig(f'./data_processing/topic modelling/exported_graphs/word_freq_{list_fields_names[field_i]}')

    ###################################################################
    ###################################################################
    ###################################################################
    # graph network

    # Create network plot
    G = nx.Graph()

    # Create connections between nodes
    for k, v in dic_top100.items():
        G.add_edge(k[0], k[1], weight=(v))

    # G.add_node("china", weight=100)

    G.edges(data=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    pos = nx.spring_layout(G, k=3)

    weights = [G[u][v]['weight'] for u, v in G.edges]


    def adjust_weights(weight):
        new_weights = []
        for i in range(len(weights)):
            new_weights.append(np.min([np.round(3 * ((weights[i]) / (np.mean(weights))) - 1, 2), 4]))
            print(new_weights[-1])
        print(f"min: {np.min(new_weights)}")
        print(f"max: {np.max(new_weights)}")
        return new_weights


    weights_adjusted = adjust_weights(weights)

    # Plot networks
    nx.draw_networkx(G, pos,
                     font_size=16,
                     width=weights_adjusted,
                     edge_color='grey',
                     node_color='#78B4CD',
                     with_labels=False,
                     ax=ax,
                     alpha=0.7)

    # Create labels
    for key, value in pos.items():
        x, y = value[0] + .05, value[1] + .05
        ax.text(x, y,
                s=key,
                bbox=dict(facecolor='orange', alpha=0.2),
                horizontalalignment='center', fontsize=13)
    plt.title(f"Most common word co-occurrences for {list_fields_names[field_i]}", size=16)
    # plt.show()
    plt.savefig(f'./data_processing/topic modelling/exported_graphs/most_common_word_co-occurrence_{list_fields_names[field_i]}')


###################################################################
###################################################################
###################################################################

# Q1. which expectations are common between different fields?


def find_similar_top_words(prob_A, prob_B):
    prob_B = prob_B.sort_values(['word_1_abs_freq', 'count'], ascending=False)

    filter_A = keep_top_n(prob_A, n=100)
    filter_A = filter_A[filter_A['word_1'] != 'sos']
    filter_A = filter_A[filter_A['word_2'] != 'eos']
    filter_A = filter_A[filter_A['count'] > 1]

    filter_B = keep_top_n(prob_B, n=100)
    filter_B = filter_B[filter_B['word_1'] != 'sos']
    filter_B = filter_B[filter_B['word_2'] != 'eos']
    filter_B = filter_B[filter_B['count'] > 1]

    np.unique(filter_A['word_1'])
    np.unique(filter_B['word_1'])

    common_top_words = []
    for each in np.unique(filter_A['word_1']):
        if each in np.unique(filter_B['word_1']):
            common_top_words.append(each)
    print("Percentage similar top words: ")
    print(len(common_top_words) / len(np.unique(filter_B['word_1'])))

    df_common_words = pd.DataFrame(columns=['word_1', 'word_2', 'prob_A', 'prob_B'])
    dic_common_words = {}
    for word_1 in common_top_words:
        # print(word_1)
        prob_A_w2 = list(prob_A[(prob_A['word_1'] == word_1) & (prob_A['count'] > 1)]['word_2'])
        prob_B_w2 = list(prob_B[(prob_B['word_1'] == word_1) & (prob_B['count'] > 1)]['word_2'])
        list_common_words = []
        dic_common_words[word_1] = []
        for word in prob_A_w2:
            if word in prob_B_w2:
                # print(word)
                if word not in list_common_words:
                    list_common_words.append(word)
                    dic_common_words[word_1].append(word)
                    df_common_words = df_common_words.append({'word_1': word_1,
                                                              'word_2': word,
                                                              'prob_A': list(prob_A[(prob_A['word_1'] == word_1) & (
                                                                      prob_A['word_2'] == word)]['cond_prob'])[0],
                                                              'prob_B': list(prob_B[(prob_B['word_1'] == word_1) & (
                                                                      prob_B['word_2'] == word)]['cond_prob'])[0]},
                                                             ignore_index=True)

                df_common_words.iloc[0, :]
    df_common_words = df_common_words.sort_values(['prob_A', 'prob_B'], ascending=False, ignore_index=1)

    return df_common_words


common_words = find_similar_top_words(fin_prob_bigram, acc_prob_bigram)
common_AB = common_words[common_words['prob_A'] >= 0.01]
common_AB = common_AB[common_AB['prob_B'] >= 0.01]

typical_A = common_words[common_words['prob_A'] >= 0.01]
typical_A = typical_A[typical_A['prob_B'] <= 0.5]

unusual_A = common_words[common_words['prob_A'] <= 0.05]
unusual_A = unusual_A[unusual_A['prob_B'] <= 0.05]

# Find top common words among different fields:

print("Common words among different fields")
print('-'*100)
for field in list_bi_prob:
    common_words = find_similar_top_words(fin_prob_bigram, field)
    common_A = common_words[common_words['prob_A'] >= 0.1]
    common_A = common_A[common_A['prob_B'] >= 0.05]
    print("new field")
    print(common_A)


#########################################
#########################################
#########################################
# Q1. which expectations appear a lot in one field but not in the other?

def find_unique_top_words(prob_A, prob_B):
    prob_B = prob_B.sort_values(['word_1_abs_freq', 'count'], ascending=False)

    filter_A = keep_top_n(prob_A, n=100)
    filter_A = filter_A[filter_A['word_1'] != 'sos']
    filter_A = filter_A[filter_A['word_2'] != 'eos']
    filter_A = filter_A[filter_A['count'] > 1]

    filter_B = keep_top_n(prob_B, n=100)
    filter_B = filter_B[filter_B['word_1'] != 'sos']
    filter_B = filter_B[filter_B['word_2'] != 'eos']
    filter_B = filter_B[filter_B['count'] > 1]

    np.unique(filter_A['word_1'])
    np.unique(filter_B['word_1'])

    common_top_words = []
    for each in np.unique(filter_A['word_1']):
        if each not in np.unique(filter_B['word_1']):
            common_top_words.append(each)
    print("Percentage similar top words: ")
    print(len(common_top_words) / len(np.unique(filter_B['word_1'])))

    df_common_words = pd.DataFrame(columns=['word_1', 'word_2', 'prob_A', 'prob_B'])
    dic_common_words = {}
    for word_1 in common_top_words:
        # print(word_1)
        prob_A_w2 = list(prob_A[(prob_A['word_1'] == word_1) & (prob_A['count'] > 1)]['word_2'])
        prob_B_w2 = list(prob_B[(prob_B['word_1'] == word_1) & (prob_B['count'] > 1)]['word_2'])
        list_uncommon_words = []
        dic_common_words[word_1] = []
        for word in prob_A_w2:
            if word not in prob_B_w2:
                # print(word)
                if word not in list_uncommon_words:
                    list_uncommon_words.append(word)
                    dic_common_words[word_1].append(word)
                    df_common_words = df_common_words.append({'word_1': word_1,
                                                              'word_2': word,
                                                              'prob_A': list(prob_A[(prob_A['word_1'] == word_1) & (
                                                                      prob_A['word_2'] == word)]['cond_prob'])[0]},
                                                             ignore_index=True)

    df_common_words = df_common_words.sort_values(['prob_A'], ascending=False, ignore_index=1)

    return df_common_words


unique_words_finanz = find_unique_top_words(fin_prob_bigram, acc_prob_bigram)
unique_words_accounting = find_unique_top_words(acc_prob_bigram, fin_prob_bigram)
unique_words_mkt = find_unique_top_words(mkt_prob_bigram, fin_prob_bigram)
unqiue_words_cons = find_unique_top_words(cons_prob_bigram, fin_prob_bigram)
unqiue_words_dat = find_unique_top_words(dat_prob_bigram, fin_prob_bigram)
unqiue_words_sup = find_unique_top_words(sup_prob_bigram, fin_prob_bigram)
unqiue_words_hr = find_unique_top_words(hr_prob_bigram, fin_prob_bigram)
unqiue_words_sale = find_unique_top_words(sale_prob_bigram, fin_prob_bigram)
unqiue_words_pr = find_unique_top_words(pr_prob_bigram, fin_prob_bigram)
unique_words_proj = find_unique_top_words(proj_prob_bigram, fin_prob_bigram)


############################################

# extract terms from unique words and common words
# (next step: classify them in categories, e.g. soft skills, hard skilla)
## Q1: which terms appear fairly often in job ads? What skills do they describe?

def get_top_word_combinartions(term, list_bi_prob):
    unique_word_pairs = []
    for each in list_bi_prob:
        word_1 = each[(each['word_1'] == term) & (each['count'] > 1)].sort_values('count', ascending=0)['word_1']
        word_2 = each[(each['word_1'] == term) & (each['count'] > 1)].sort_values('count', ascending=0)['word_2']
        unique_word_pairs = unique_word_pairs + list(word_1 + "_" + word_2)
    for each in list_bi_prob:
        word_1 = each[(each['word_2'] == term) & (each['count'] > 1)].sort_values('count', ascending=0)['word_1']
        word_2 = each[(each['word_2'] == term) & (each['count'] > 1)].sort_values('count', ascending=0)['word_2']
        unique_word_pairs = unique_word_pairs + list(word_1 + "_" + word_2)
    print(np.unique(unique_word_pairs))
    return list(np.unique(unique_word_pairs))


# define dictionary with skill categories
dic_requirements = {
    'educ_promotion':[],
    'educ_master': [
        'bachelor_mast', 'bachelor_masterstudium', 'diplom_masterstudium', 'bachelor_masterniveau',
        'abgeschloss_mast', 'mast_diplom', 'bachelor_masterstudium' 'masterstudium_fachricht'
    ],
    'educ_bachelor': [
        'bachelor_mast', 'bachelor_masterstudium', 'diplom_bachelorstudium', 'bachelor_scienc', 'bachelor_or',
        'bachelor_masterniveau',
        'abschluss_bachelor',
        'bachelor_mast', 'bachelor_masterstudium', 'diplom_bachelor', 'studium_bachelor'
    ],
    'educ_ausbildung': [
        'abgeschloss_ausbild', 'ausbild_abgeschloss', 'ausbild_abschluss', 'ausbild_b2bvertriebserfahr',
        'ausbild_bankkaufmann', 'ausbild_berufserfahr', 'ausbild_entsprech', 'ausbild_erfolgreich', 'ausbild_gern',
        'ausbild_marketg', 'ausbild_mdest', 'ausbild_mehr', 'ausbild_pril', 'ausbild_steuerfachangestellt',
        'ausbild_studium', 'ausbild_theoret', 'ausbild_versicherungsbranch', 'ausbild_versicherungsfachmann',
        'ausbild_vorzugsweis', 'ausbild_weiterbild', 'dual_ausbild', 'kaufmann_ausbild', 'studium_ausbild',
        'technisch_ausbild', 'verfug_ausbild', 'vergleichbar_ausbild',
        'abgeschloss_berufsausbild', 'berufsausbild_berufserfahr', 'berufsausbild_steuerfachangestellt',
        'berufsausbild_weiterbild', 'kaufmann_berufsausbild'
    ],
    'leadership_skills': [
        'fuhrung_projektteam', 'organisationsfah_belastbar',
        'organisationsfah_ren',
        'verhlungsgeschick_umgang', 'vertrauenswurd_umgang',
        'fuhrung_terdisziplar',
        'umgang_leistungsdruck',
        'proaktiv_denkweis',
        'affitat_gestalt', 'aktiv_gestalt', 'arbeit_gestalt', 'aug_gestalt',
        'eigenverantwort_gestalt',
        'positiv_denkweis',
        'strateg_denkweis',
        'zielorientiert_denkweis',
        'unternehmer_denkweis',
        'begeisterungsfah_routiert', 'begeisterungsfah_selbstbestmt',
        'uberzeugt_ebenso', 'proaktiv_arbeitsweis', 'proaktiv_gierig',
        'proaktiv_kommunikation', 'proaktiv_wissensmanagement', 'proaktiv_arbeit',
        'belastbar_durchsetzungsfah', 'durchsetzungsfah_grsatzlich',
        'durchsetzungsfah_ausgepragt', 'durchsetzungsfah_ren', 'durchsetzungsfah_eos',
        'belastbar_sd',
        'belastbar_eigenitiativ',
        'belastbar_uberdurchschnitt',
        'belastbar_teamgeist',
        'belastbar_teamfah',
        'belastbar_sich',
        'belastbar_zuverlass',
        'belastbar_fahig',
        'belastbar_selbstand',
        'belastbar_kommunikationsstark',
        'beruf_fuhrungserfahr',
        'denkvermog_umsetzungsstark',
        'fuhrungsperson_entscheidungsstark',
        'entscheidungsstark_eigenitiativ',
        'kommunikativ_verhlungsgeschick',
        'leitung_teilprojekt',
        'stark_uberzeugungsfah',
        'souveran_auftret',
        'bereit_fuhrungsqualitat',
        'leitung_projektteam',
        'ausgepragt_durchsetzungsvermog',
        'verhandlungssicher',
        'kompetenz_durchsetzungsfah',
        'fahig_ubersteugungsstark',
        'ubersteugungsstark_zielorientiert',
        'verhandlungsstark',
        'motivation_teamgeist',
        'geschaftsstrategi_fahig',
        'projekt_teamleitungserfahr',
        'verhlungsstark_fahig',
        'organisator_fahig',
        'hsonmentalitat_fahig',
        'kommunikationsstark_fahig',
        'fahig_durchsetzungsvermog',
        'fahig_ubersteugungsstark',
        'fahig_problemlosungskompetenz',
        'unternehmer_denkweis',
    ],
    'soft_skills_communication_team': [
        'teamgeist_kommunikationsstark',
        'umgang_mensch',
        'schreibtalent_gespur',
        'kommunikationsstark_eigenitiativ',
        'kommunikationsstark_deutsch', 'kommunikationsstark_teamfah', 'kommunikationsstark_hierarchieeb',
        'kommunikationsstark_teamplay',
        'grupp_umgang', 'team_grupp',
        'gespur_sprach', 'gespur_storytellg',
        'gespur_kenbedurfnis',
        'gespur_kenwunsch',
        'flexibilitat_teamfah',
        'sprachgefuhl_talent', 'sprachlich_talent',
        'talent_begeister',
        'talent_schreib',
        'kommunikation_teamfah',
        'arbeitsweis_empath', 'empath_auftret', 'empath_managementeb', 'empath_person', 'empath_serviceorientiert',
        'empath_sozialkompetent', 'frelich_empath', 'kommunikationsstark_empath', 'kommunikativ_empath',
        'konzeptionsfah_empath', 'kreativ_empath', 'sich_empath', 'team_empath', 'uberzeug_empath',
        'kontaktfreud_kommunikationsstark', 'kontaktfreud_reis', 'kontaktfreud_spass',
        'kommunikationsstark_analyt', 'kommunikationsstark_ausgepragt', 'kommunikationsstark_begeisterungsfah',
        'kommunikationsstark_deutsch', 'kommunikationsstark_durchsetzungsvermog', 'kommunikationsstark_eigenitiativ',
        'kommunikationsstark_fahig', 'kommunikationsstark_flexibel', 'kommunikationsstark_frelich',
        'kommunikationsstark_gut', 'kommunikationsstark_hoh', 'kommunikationsstark_kenorientiert',
        'kommunikationsstark_kreativitat', 'kommunikationsstark_lebend', 'kommunikationsstark_leidenschaft',
        'kommunikationsstark_pass', 'kommunikationsstark_person', 'kommunikationsstark_reisebereitschaft',
        'kommunikationsstark_ren', 'kommunikationsstark_selbstbewusst', 'kommunikationsstark_selbststand',
        'kommunikationsstark_sich', 'kommunikationsstark_sowohl', 'kommunikationsstark_spass',
        'kommunikationsstark_sprachkenntnis', 'kommunikationsstark_teamfah', 'kommunikationsstark_teamplay',
        'kommunikationsstark_teamplayermentalitat', 'kommunikationsstark_verhlungsgeschick',
        'kommunikationsstark_verhlungssich', 'kommunikationsstark_versteh',
        'kontaktfreud_kommunikationsstark', 'kontaktfreud_reis', 'kontaktfreud_spass',
        'kenntnis_teamwork', 'kenntnis_zusammenarbeit',
        'kommunikationsstark_kenorientiert', 'kommunikationsstark_ren', 'kommunikationsstark_reisebereitschaft',
        'kommunikationsstark_teamplayermentalitat', 'kommunikationsstark_verhlungssich',
        'kommunikationsstark_versteh', 'kommunikationsstark_flexibel', 'kommunikationsstark_freud',
        'kommunikationsstark_flexibilitat', 'kommunikationsstark_teamorientier',
        'kommunikationsstark_serviceorientiert',
        'kommunikationsstark_kreativitat', 'kommunikationsstark_lebend', 'kommunikationsstark_durchsetzungsvermog',
        'kommunikationsstark_person', 'kommunikationsstark_selbststand', 'kommunikationsstark_fahig',
        'kommunikationsstark_hoh', 'kommunikationsstark_wort', 'kommunikationsstark_uberzeug',
        'kommunikationsstark_sich',
        'teamfah_verantwortungsvoll',
        'zusammenzuarbeit_kommunikationsstark',
        'kultur_zusammenzuarbeit',
        'verschied_mensch',
        'positiv_ausstrahl',
        'ansteck_positiv',
        'positiv_estell', 'positiv_optist', 'positiv_ausstrahl', 'positiv_uberzeug', 'positiv_sympath',
        'positiv_optist', 'positiv_souveran',
        'direkt_zusammenarbeit',
        'kontaktfreud_kommunizi',
        'kommunikationsstark_uberzeug',
        'uberzeug_mensch',
        'team_wohl', 'team_begeist', 'team_est', 'team_kommunikationsfah',
        'team_besitzt', 'team_kommunikationsstark', 'team_tegri',
        'kommunikationsstark_selbstbewusst',
        'berat_sozialkompetenz',
        'team_spass', 'team_mentor', 'team_projektarbeit', 'team_uberzeugungsfah',
        'leistung_teamarbeit',
        'umgang_mensch',
        'kommunikativ_art',
        'ausgezeichnet_kommunikationsfah',
        'kommunikationsfah_hsonmentalitat',
        'eigenitiativ_kommunikationsgeschick',
        'uberaus_kommunikativ',
        'ausgepragt_teamgefuhl',
        'sich_auftret',
        'ternational_teamplay',
        'exzellent_kommunikationsfah',
        'kommunikationsfah_deutsch',
        'kommunikationsfah_flexibilitat',
        'kommunikationsstark_teamplay',
        'gut_sprachgefuhl',
        'leidenschaft_storytellg',
        'storytellg_verfass',
        'kreativ_kommunikationsstark',
        'kommunikationsstark_person',
        'kommunikationstalent_gut',
        'gut_verhlungsgeschick',
        'uberzeug_kommunikation',
        'empath_person',
        'gern_team',
        'teamfah_kommunikationsstark',
        'kontakt_mensch',
        'ausgepragt_kommunikationsstark',
        'verantwortungsbewussts_kommunikationsfah',
        'hervorrag_kommunikationsfah',
        'stark_kommunikationsfah',
        'team_kommunikationsfah',
        'ausgepragt_kommunikationsfah',
        'kommunikation_prasentationsfah',
        'teamgeist_kommunikation',
        'hoh_kenorientier',
        'stark_kommunikationsvermog',
        'freud_zusammenarbeit',
        'arbeit_teamorientiert',
        'sozial_kompetenz',
        'teamfah_person',
        'uberzeug_auftret',
        'stark_kommunikationsvermog'
        'ausgepragt_kommunikationsfah',
        'kommunikationsfah_strukturiert',
        'hoh_kommunikation',
        'kommunikativ_person',
        'freud_zusammenarbeit',
        'terpersonell_fahig',
        'kommunizierst_exzellent',
        'team_organisi',
        'freud_teamarbeit',
        'hoh_sozialkompetenz',
        'engagiert_teamplay',
        'kommunikationsgeschick',
        'verantwortungsbewussts_kommunikationsfah',
        'team_affitat',
        'team_begeist',
        'team_arbeit',
        'kommunikativ_fahig',
        'rhetor_fahig',
        'kommunikation_teraktionsfah',
        'kommunikation_prasentationsfah',
        'kommunikation_kooperationsstark',
        'kommunikation_teamfah',
        'kommunikation_sozialkompetenz',
        'kommunikation_prasentationsstark',
        'ausgezeichnet_kommunikation',
        'schriftlich_kommunikation',
        'ausgepragt_kommunikation'
        'kommunikativ_engagiert'
    ],
    'soft_skills_resilience':[
        'anpassungsfah_hoh', 'anpassungsfah_kontext', 'flexibilitat_anpassungsfah', 'flexibl_anpassungsfah',
        'schnell_anpassungsfah',
        'flexibilitat_teamfah',
        'umgang_leistungsdruck',
        'eigenitiativ_selbststand',
        'denkvermog_selbststand', 'denkweis_selbststand', 'durchsetzungsvermog_selbststand',
        'belastbar_selbststand', 'bereit_selbststand', 'bereitschaft_selbststand',
        'betreu_selbststand', 'dabei_selbststand', 'darzustell_selbststand', 'datev_selbststand',
        'leistungsdruck_kee', 'leistungsdruck_saub',
        'auftret_selbststand', 'ausgesproch_selbststand',
        'erfolg_leist',  'leist_eigenverantwort',
        'uberdurchschnitt_leist',
        'dienstleistungsbereitschaft_leistungsbereitschaft',
        'ausgepragt_leistungsbereitschaft',
        'arbeit_belastbar', 'arbeitsweis_belastbar', 'auftret_belastbar', 'belastbar_analyt', 'belastbar_arbeit',
        'belastbar_ausgepragt', 'belastbar_bereitschaft', 'belastbar_berufserfahr', 'belastbar_besitzt',
        'belastbar_durchsetzungsvermog', 'belastbar_effizient', 'belastbar_eigenitiativ', 'belastbar_eigenverantwort',
        'belastbar_engagement', 'belastbar_englischkenntnis', 'belastbar_eos',
        'belastbar_esatzbereitschaft', 'belastbar_exzellent', 'belastbar_fahig',
        'belastbar_fdest', 'belastbar_flexibel', 'belastbar_flexibilitat',
        'belastbar_fliessend', 'belastbar_freud', 'belastbar_genau',
        'belastbar_gut', 'belastbar_hoh', 'belastbar_identifikationsbereitschaft',
        'belastbar_kommunikationsfah', 'belastbar_kommunikationsstark',
        'belastbar_leistungsbereitschaft', 'belastbar_losungsorientier',
        'belastbar_losungsorientiert', 'belastbar_moglich', 'belastbar_ms',
        'belastbar_organisationsgeschick', 'belastbar_organisationsstark',
        'belastbar_organisationstalent', 'belastbar_organisiert', 'belastbar_person',
        'belastbar_prnetzwerk', 'belastbar_reisebereitschaft', 'belastbar_ren',
        'belastbar_schwierig', 'belastbar_sd', 'belastbar_selbstand',
        'belastbar_selbstorganisation', 'belastbar_selbststand', 'belastbar_sich',
        'belastbar_sorgfalt', 'belastbar_strukturiert', 'belastbar_teamfah',
        'belastbar_teamgeist', 'belastbar_teamplay', 'belastbar_teressiert',
        'belastbar_unterstutz', 'belastbar_vergut', 'belastbar_verlierst',
        'belastbar_zeichn', 'belastbar_zeitdruck', 'belastbar_zielorientiert',
        'belastbar_zuverlass', 'durchsetzungsvermog_belastbar', 'eigenitiativ_belastbar',
        'eigenmotivation_belastbar', 'engagement_belastbar', 'ergebnisorientier_belastbar',
        'esatzbereitschaft_belastbar', 'flexibel_belastbar', 'flexibilitat_belastbar',
        'flexibl_belastbar', 'gepaart_belastbar', 'hoh_belastbar', 'hweg_belastbar',
        'kenorientier_belastbar', 'kommunikationsfah_belastbar', 'kommunikationsstark_belastbar',
        'kommunikativ_belastbar', 'kreativitat_belastbar', 'leistungsbereitschaft_belastbar',
        'lernbereitschaft_belastbar', 'mass_belastbar', 'mentalitat_belastbar', 'motivation_belastbar',
        'organisationstalent_belastbar', 'schreib_belastbar', 'schrift_belastbar', 'sd_belastbar',
        'sorgfalt_belastbar', 'stressresistent_belastbar', 'teamfah_belastbar', 'teamgedank_belastbar',
        'teamgeist_belastbar', 'teamorientiert_belastbar', 'uberzeugungsfah_belastbar',
        'verantwortungsbewussts_belastbar', 'zielorientier_belastbar', 'zuverlass_belastbar'

    ],
    'soft_skills_innovation_digital': [
        'digital_affitat',
        'freud_novativ',
        'gespur_aktuell', 'gespur_bewerbermarkt', 'gespur_digital',
        'gespur_mark', 'gespur_markt', 'gespur_markttrend',
        'kreativitat_novationsgedank', 'novationsgedank_organisationstalent',
        'teress_digital',
        'novativ_denkweis',
        'teressierst_digital',
        'begeist_digital', 'begeister_digital', 'begeisterst_digital',
        'entwickl_novativ',
        'brenn_digital',
        'kreativ_novativ', 'novativ_denk', 'novativ_denkweis','novativ_person',
        'novativ_technisch', 'novativ_technologi', 'novativ_them', 'trend_novativ', 'umgang_novativ', 'umsetz_novativ',
    'zukunftstracht_novativ',
        'digital_trend',
        'anwend_digital', 'arbeit_digital', 'begeist_digital', 'begeister_digital', 'begeisterst_digital',
        'begeister_novativ',
        'digital_nativ', 'digital_novation', 'digital_novator',  'verstandnis_digital', 'versteh_digital',
        'verwalt_digital', 'zeitgemass_digital',
        'novator_unternehm', 'novator_visio',
        'auszuprobi_novativ',
        'begeisterst_novativ',
        'affitat_it',
        'begeister_technik',
        'begeisterst_technisch',
        'novationsfreud_begeisterst',
        'teress_digitalisier',
        'technisch_erung',
        'trend_technologi',
        'erung_novativ',
        'them_digitalisier',
        'itlos_fragestell',
        'technisch_affitat',
        'hoh_itaffitat',
        'verstandnis_technologietr',
        'leidenschaft_digital',
        'begeister_novativ',
        'ittechn_verstandnis',
        'digitalisier_geschaftsprozess',
        'affitat_technologi',
        'technologi_digitalisierungsthem',
        'digital_novator',
        'digital_medienntkontaktfreud',
        'digital_tragsformat',
        'digital_transformation',
        'affitat_technik',
        'novativ_teamplay', 'novativ_losung',
        'affitat_zahl',
        'affitat_itkommunikation',
        'affitat_analys',
        'tchnisch_affitat',
        'technologi_kommunikation',
        'technologi_begeisterst',
        'technologi_architektur',
        'technologi_trend',
        'technologi_knowhow',
        'technologi_fuhrungskompetenz',
        'technologi_begeist',
        'technologi_agil',
        'technologi_digitalisierungsthemen',
        'technologi_entwickl',
        'technologi_bereitschaft',
        'technologi_ambition',
        'novativ_losung ',
        'novativ_logistikprozess',
        'novativ_umfeld',
        'aktuell_technologi',
        'aktuell_digitalisierungsthem',
        'begeister_technologi',
        'novativ_technologi ',
        'technologi_digitalisier',
        'novativ_kopf',
        'technologi_digitalisierungsthem',
        'novativ_denk',
        'umsetz_novativ',
        'teress_novativ',
        'novativ_kreativ',
        'gespur_digital',
        'novativ_digitalisierungsthem'
    ],
    'travel': [
        'bereit_reis',
        'reis_vorteil',
        'teamarbeit_reisetat',
        'dienstreis_deutschl',
        'dienstreis_eos',
        'national_reis',
        'reisetat_flexibiliat',
        'reisetat_deutsch',
        'reisetat_earbeitungsprogramm',
        'hoh_reisebereitschaft',
        'ternational_reisebereitschaft',
        'bereitschaft_dienstreis',
        'weltweit_dienstreis',
        'bereitschaft_reis',
        'gelegent_dienstreis',
        'ternational_reisetat',
        'bereitschaft_reisetat',
        'ntprojektbedgt_reisebereitschaftnn',
        'reisebereitschaft_flexibilitat',
        'flexibilitat_belastbar',
        'belastbar_eos',
        'hoh_eigenmotivation',
        'regional_reisebereitschaft',
        'reisebereitschaft_arbeit'
    ],
    'denkweis_analytisch': [
        'analyt_konzeptionell',
        'analyt_zusammenhang',
        'gespur_zielgruppengerecht', 'gespur_zielgruppenspezif', 'gespur_zusammenhang',
        'gefuhl_zahl',
        'denkweis_datenbasiert',
        'systemat_vorgeh',
        'denk_strukturiert',
        "zahl_faktenbasiert",
        "zahl_strukturiert",
        'schnell_verstand',
        'fahig_zahl',
        'analyt_stet',
        'stet_losungsorientiert',
        'analyt_fragestell',
        'zusammenhang_analyt',
        'analyt_strukturiert',
        'analyt_denkvermog',
        'hsonmentalitat_analyt',
        'affitat_kennzahl',
        'komplex_zusammenhang',
        'selbstand_strukturiert',
        'analyt_denkweis',
        'analysierst_problem',
        'denkweis_pragmat',
        'analyt_konzeptionell',
        'analyt_verstandnis',
        'analyt_denkvermog',
        'analyt_denk',
        'analyt_strukturiert',
        'analyt_kompetenz',
        'analyt_kommunikativ',
        'analyt_konzeptionell',
        'analyt_zusammenhang',
        'analyt_durchdrg',
        'analyt_geschickt',
        'analyt_konzeptionell',
        'losungsorientiert_analyt',
        'stark_analys',
        'wirtschafts_denkweis',
        'strukturiert_denkweis',
        'zielorientiert_denkweis',
        'losungsorientiert_denkweis',
        'analytics_fahig',
        'arbeit_datenbezog',
        'ausgepragt_analyt',
        'esatz_data',
        'wunschenswert_analyt',
        'analyt_fahig',
        'verstandnis_analyt'
    ],
    'denkweis_kreativ': [
        'kreativ_denkweis',
        'asthet_gespur',
        'gespur_empathi',
        'gespur_farb', 'gespur_fashion', 'gespur_gestalt',  'gespur_grafik', 'gespur_grafisch','gespur_layout',
        'visuell_gespur',
        'gestalter_gespur',
        'kreativ_gespur', 'kreativitat_gespur',
        'gespur_farb', 'gespur_fashion',
        'gespur_design',
        "gespur_trend",
        'kreativitat_hoh',
        'arbeitsweis_kreativitat',
        "gespur_aktuell",
        "kreativitat_kommunikationsstark",
        "kreativitat_gespur",
        'mass_kreativitat',
        'kreativ_arbeit',
        'kreativ_arbeit', 'kreativ_denk', 'kreativ_ide', 'kreativ_kommunikationsstark', 'kreativ_losungsorientiert',
        'kreativ_novativ', 'kreativ_team', 'kreativ_teamfah', 'kreativ_text',
        'novativ_kreativ',
        'gespur_kreativ',
        'novativ_denkweis',
        'kreativ_herangehensweis',
        'novativ_fahig',
        'kreativ_los',
        'kreativ_denk',
        'kreativ_losungsansatz',
        'kreativ_entwickeln',
        'kreativ_selbststand',
        'kreativ_ide',
        'hoh_kreativitat',
    ],
    'pc_basic': [
        'datev_allgem',
        'umgang_pc',
        'umgang_googl',
        'anwend_gangig', 'anwenderkenntnis_gangig', 'arbeit_gangig', 'bedi_gangig', 'beherrsch_gangig', 'berufserfahr_gangig', 'bildbearbeit_gangig',
        'datev_gangig',
        'umgang_dat',
        'umgang_datenbank',
        'umgang_digital',
        'umgang_edv', 'umgang_edvprogramm',
        'umgang_ms', 'umgang_msexcel', 'umgang_msfic', 'umgang_msficeanwend', 'umgang_msficepaket',
        'umgang_msficeprodukt', 'umgang_msficeproduktennteigenverantwort', 'umgang_msficeprogramm',
        'umgang_datevprodukt', 'umgang_datevprogramm',
        'pc_gangig', 'msficeprogramm_gangig', 'gangig_technologi',
        'gangig_ms','gangig_msfic', 'gangig_msficeanwend', 'gangig_msficeprodukt',
        'edvkenntnis_gangig',
        'kenntnis_microst',
        'powerpot_prasenti',
        'msficekenntnis_englischkenntnis',
        'umgang_edvsystem',
        'edvsystem_vorteil',
        'msfic_anwend',
        'umgang_msficepaket',
        'kenntnis_msficeanwend',
        'msficekenntnis_englischkenntnis',
        'edvkenntnis_ficeanwend', 'edvkenntnis_msfic', 'edvkenntnis_sbesond', 'edvkenntnis_sich',
        'umgang_kennzahl',
        'umgang_pc',
        'kenntnis_msexcel',
        'umgang_msficeprogramm',
        'ms_access',
        'umgang_msfic',
        'ficeprogramm_excel',
        'speziell_excel',
        'sb_excel',
        'umgang_excel',
        'outlook_excel',
        'microst_excel',
        'kenntnis_excel',
        'sbesond_excel',
        'outlook_excel', 'paket_excel', 'pot_excel', 'powerpot_excel', 'programm_excel', 'project_excel',
        'word_excel ',
        'teamplay_excel',
        'excel_powerpot',
        'grlegend_ficekenntnis',
        'gangig_msficeanwend',
        'pow_pot',
        'gangig_msficeprodukt',
        'msficeprodukt_besond',
        'besond_excel',
        'gangig_msfic',
        'msficeanwend_excel',
        'stardmsfic_programm',
        'kenntnis_msfic',
        'kenntnis_datev',
        'gangig_msficeprogramm',
        'microst_excel',
        'allgeme_itkenntnis',
        'kenntnis_ms',
        'fiert_msficekenntnis',
        'kenntnis_excel',
        'excel_sd',
        'wunschenswert_msficekenntnis',
        'stard_msficeanwend',
        'ms_365',
        'excelkenntnis_generell',
        'technisch_verstandnis',
        'datenbank_verstandnis',
        'umgang_datev',
        'umgang_ms',
        'kenntnis_datev'
        'datev_ms',
        'googl_docs',
        'googl_driv',
        'googl_calendar',
        'googl_alibaba',
        'ms_fic',
        'ms_excel',
        'ms_ficekenntnis',
        'ms_ficeprodukt',
        'ms_pow',
        'ms_crm',
        'ms_ficeanwend',
        'ms_ficepaket',
        'ms_ficeprogramm',
        'ms_dynamics',
        'umgang_datenbank',
        'sql_umgang',
        'cloud_plattform',
        'cloud_dustri',
        'data_cloud',
        'data_driv',
        'pc_kommunikation',
        'sbesond_excel',
        'ms_excel',
        'word_excel',
        'outlook_excel',
        'ficeprogramm_excel',
        'excel_arbeit', 'excel_ausgepragt',
        'speziell_excel',
        'sb_excel',
        'excel_microst',
        'excel_prazis',
        'excel_praktisch',
        'excel_pivottabell',
        'excel_kenn', 'excel_kenntnis', 'excel_kenntnissebevorzugt', 'excel_klusiv',
        'excel_deutsch', 'excel_deutschkenntnis',
        'excel_brgst', 'excel_crm', 'excel_datenbank',
        'microst_excel','besond_excel',
        'excel_analyt',
        'umgang_excel',
        'besitz_excel',
        'umgang_bildbearbeitungsprogramm',
        'ms_ficepaket', 'ms_ficeprodukt', 'ms_ficeprogramm',
        'anwend_ms', 'anwenderkenntnis_ms', 'vertraut_ms','ms_outlook', 'ms_pow', 'ms_powerpot', 'ms_project', 'ms_sharepot',
        'ms_word',
        'outlook_ms','pc_ms', 'pckenntnis_ms', 'powerpot_ms',
        'selbstverstand_ms',
        'stwareprodukt_ms',
        'verfug_ms', 'vertraut_ms', 'vorteil_ms', 'vorzugsweis_ms'
    ],
    'pc_specialized': [
        'sql_tableau',
        'grafik_stwar',
        'umgang_modellierungstechn',
        'umgang_grafik', 'umgang_grafikprogramm',
        'umgang_html',
        'umgang_salesforc',
        'umgang_sap', 'umgang_sapfi',
        'umgang_steuerprogramm',
        'umgang_web', 'umgang_webanalysetool',
        'umgang_warenwirtschaftssyst', 'umgang_warenwirtschaftssystem',
        'umgang_shopsyst', 'umgang_shopwar',
        'umgang_erpsyst', 'umgang_erpsystem',
        'admistration_sap', 'aktuell_sap', 'anwend_sap', 'anwenderkenntnis_sap', 'aws_sap',
        'datev_sap',
        'basis_sap', 'beispiel_sap', 'beispielsweis_sap', 'bereich_sap', 'bereit_sap', 'berufserfahr_sap', 'bevorzugt_sap', 'bi_sap', 'bo_sap', 'buchhaltungsstwar_sap', 'buchhaltungsstwaretool_sap', 'buchhaltungssyst_sap', 'bw_sap', 'cloud_sap', 'co_sap', 'consultant_sap', 'corefunktion_sap', 'crm_sap', 'datev_sap', 'efuhr_sap', 'englischkenntnis_sap', 'entwickl_sap', 'erfahr_sap', 'erp_sap', 'erpstwar_sap', 'erpsyst_sap', 'erpsystem_sap', 'esatz_sap', 'especially_sap', 'excel_sap', 'fi_sap', 'fic_sap', 'ficekenntnis_sap', 'ficepaket_sap', 'ficeprodukt_sap', 'ficeumgeb_sap',
        'fiert_sap', 'fokus_sap', 'grkenntnis_sap', 'hana_sap', 'hbuch_sap', 'hgb_sap', 'hr_sap', 'hybris_sap', 'ifrskenntnis_sap', 'isu_sap', 'itlos_sap', 'kenauftrag_sap', 'kenntnis_sap', 'kostenrechn_sap', 'kreditorenbuchhaltungperiodenabschlussarbeit_sap', 'les_sap', 'lohnabrechnungssystem_sap', 'losung_sap', 'msfic_sap', 'msficeanwend_sap', 'msficepaket_sap', 'nutzung_sap', 'object_sap', 'oracl_sap', 'planung_sap', 'plattform_sap', 'plementierungsberat_sap', 'plementierungskenntnis_sap', 'powerpot_sap', 'praxis_sap', 'programm_sap', 'projekterfahr_sap', 'prozess_sap',
        'datev_googl',
        'umgang_adobeprogramm',
        'adsmanagement_tool',
        'kenntnis_adsmanagement',
        'computerwissenschaft_datenstruktur',
        'datenstruktur_datenbank',
        'datenbank_postgresql',
        'postgresql_mysql',
        'kampagnenmanagementtool_sas',
        'sas_analys',
        'kenntnis_adob',
        'photoshop_kenntnis',
        'mysql_webanwend',
        'webanwend_erfahr',
        'umgang_grafikprogramm',
        'grafikprogramm_cmssyst',
        'cmssyst_joomla',
        'erfahr_youtub',
        'wordpress_adob',
        'typo_erfahr',
        'grlegend_typo',
        'adob_creativ',
        'onl_plattform',
        'msficeanwend_sap',
        'kenntnis_statist',
        'werkzeug_homepagegestalt',
        'homepagegestalt_html',
        'html_php',
        'cms_typo3',
        'typo3_webanalysetool',
        'php_solid',
        'data_management',
        'html_analyt', 'html_css', 'html_csskenntnis', 'html_erhalt', 'html_kenntnis',
        'photoshop_design', 'photoshop_illustrator',
        'adob_analytics', 'adob_cc', 'adob_creativ', 'adob_cs', 'adob_photoshop', 'adob_suit',
        'cloud_architecturehsonexpertis', 'cloud_architektur', 'cloud_b2c', 'cloud_computg', 'cloud_engeerg',
        'cloud_for', 'cloud_geseh', 'cloud_marketg', 'cloud_plattform', 'cloud_sap', 'cloud_servic', 'cloud_setz',
        'cloud_sich', 'cloud_solution', 'cloud_wunschenswert',
        'typo3_googl', 'typo3_sd', 'typo3_wordpress',
        'ms_dynamics', 'wordpress_ausgepragt', 'wordpress_googl',
        'ms_azur',
        'kenntnis_haccp',
        'datenanalysetool_sd',
        'msexcel_sap',
        'sbesond_typo3',
        'typo3_relevant',
        'framework_saf',
        'data_studio',
        'basis_sap',
        'sap_isu',
        'bzw_4hana',
        '4hana_umfeld',
        'umfeld_javabasisert', 'java_bachelor', 'java_javascript', 'java_kenntnis', 'java_net', 'java_ntfreud',
        'java_shell', 'java_sprg', 'java_web',
        'javabasisert_applikation',
        'verschied_cms',
        'googleprodukt_analytics',
        'thkg_devops',
        'sap_fi',
        'it_management',
        'gross_datenmeng',
        'relational_datenbank',
        '4hana_sfdc',
        'excel_sap',
        'sap_r3',
        'excel_sapkenntnis',
        'umgang_lux',
        'datenbanktechnologi_kenntnis',
        'kenntnis_queugtechn',
        'fic_sap',
        'salesforc_marketg',
        'etabliert_erpsystem',
        'msficeanwend_vorausgesetzt',
        'sag_msficeanwend',
        'umgang_erpsystem',
        'umgang_sap',
        'anwend_photoshop',
        'photoshop_design',
        'design_illustrator',
        'illustrator_premi',
        'sap_abap',
        'cloud_computg',
        'big_data',
        'dustri_40',
        'sapkenntnis_wunschenswert',
        'sap_bw',
        'certified_syst',
        'web_content',
        'kenntnis_web',
        'losung_ams',
        'erfahr_sap',
        'saas_datenbank',
        'sqlabfrag_auswertungenerstell',
        'db2umfeld_sqlabfrag',
        'azur_gcp',
        'azur_aws',
        'azur_amazon',
        'public_cloud',
        'cloud_frastruktur',
        'crmsystem_sapkenntnis',
        'secaas_cloud',
        'cloud_security',
        'frastruktur_networksecurity',
        'kenntnis_sicherheitsstard',
        'gcp_frastruktur',
        'azur_sd',
        'azur_dien',
        'technologi_sap',
        'technologi_azur',
        'datev_msfic',
        'googl_analytics',
        'googl_ads',
        'googl_seo',
        'googl_data',
        'googl_merchant',
        'googl_search',
        'googl_shoppg',
        'googl_marketg',
        'googl_adword',
        'googl_youtub',
        'sap_sd',
        'sap_hana',
        'sap_custom',
        'sap_kenntnis',
        'sap_abap',
        'sap_crm',
        'sap_coremodul',
        'sap_4hana',
        'sap_vertrieb',
        'sap_crm',
        'sap_commerc',
        'sap_fiori',
        'sap_analytics',
        'sap_fscm',
        'sap_r3',
        'sap_erp',
        'sap_cloud',
        'sap_4hanaumfeld'
        'sap_fi',
        'sap_modul',
        'sap_saleforc',
        'sap_marketg',
        'cloud_computg',
        'cloud_sap',
        'azur_cloud',
        'hana_cloud',
        'object_cloud',
        'data_cent',
        'data_blockcha',
        'data_warehous',
        'tool_tableau',
        'tool_googl',
        'data_studio',
        'pow_bi',
        'aws_salesforc',
        'amazon_web',
        'cloud_solution',
        'javascript_ampscript',
        'ampscript_sql',
        'sql_html5',
        'html5_css3',
        'nutzung_git',
        'nod_js',
        'apex_admistration',
        'admistration_web',
        'web_apps',
        'data_warehous', 'data_analyst', 'data_governanc', 'data_engeerg', 'data_mg', 'data_scientist', 'data_cent',
        'data_driv', 'data_pipel', 'data_cloud', 'data_lak', 'data_visualizationempath', 'data_anwend', 'data_robot',
        'data_studio', 'data_observation', 'data_analys', 'data_evaluation', 'data_technologi',
        'data_visualisier', 'data_vault', 'data_tegration', 'data_solution', 'data_set', 'data_sourc', 'data_expert',
        'data_schem',
        'data_provid', 'data_stardization', 'data_sourcg', 'data_relational', 'data_warehousg', 'data_bas',
        'data_protection',
        'java_kenntnis', 'java_shell', 'java_net', 'java_go', 'java_bachelor', 'java_ntfreud',
        'java_sprg', 'java_javascript',
        'python_java', 'python_bzw', 'python_or', 'python_dock', 'python_googl', 'python_deployment',
        'datenbank_kenntnis', 'datenbank_sap', 'datenbank_sd', 'datenbank_sql',
        'datenbank_netzwerk', 'datenbank_webtechnologi', 'datenbank_sd', 'datenbank_verstandnis', 'datenbank_crm',
        'datenbank_tool',
        'sap_crm', 'sap_kenntnis', 'sap_coremodul', 'sap_custom', 'sap_r3', 'sap_gut', 'sap_fi', 'sap_co',
        'sap_vorausgesetzt', 'sap_modul', 'sap_vorteil',
        'sap_sal', 'sap_servic', 'sap_marketg', 'sap_commerc'
    ],
    'programming_languages': ['programmiersprach_datenbank',
                              'framework_tensorflow',

                              'erstell_hardwar', 'hardwar_branch', 'hardwar_netzwerk', 'hardwar_pcs', 'hardwar_stwarekonfiguration',
                              'halbleit_stwar', 'hard_stwar',
                              'analysis_python', 'anwend_python', 'cd_python', 'datenbankkenntnis_python',
                              'erfahr_python', 'java_python', 'javascript_python', 'kenntnis_python', 'net_python',
                              'ntnutzung_python', 'ntprojekterfahr_python', 'programmiersprach_python',
                              'python_access', 'python_bzw', 'python_deployment', 'python_erfahr', 'python_etc',
                              'python_go', 'python_gut', 'python_java', 'python_javascript', 'python_kenntnis',
                              'python_kne', 'python_matlab',
                              'python_sas', 'python_scala', 'python_sql', 'sprach_python', 'sql_python',
                              'tool_python', 'typescript_python',
                              'java_kenntnis', 'kenntnis_programmiersprach', 'kenntnis_itsystem',
                              'kenntnis_spass',
                              'erfahr_stwarevalidierungsstrategi',
                              'wiss_computerwissenschaft',
                              'kenntnis_bank', 'kenntnis_db',
                              'kenntnis_stwareentwickl',
                              'eschlag_cadstwar',
                              'html_referenz', 'html_kenntnis', 'html_css', 'html_analyt', 'html_referenz',
                              'html_typo3', 'html_csskenntnis',
                              'programmiersprach_python',
                              'programmiersprach_java',
                              'peripheri_stwarekenntnis',
                              'tool_python',
                              'stwar_architekturpatt',
                              'kenntnis_stwareentwickl',
                              'stwar_zeitgemass',
                              'stwar_engeerg',
                              'stwar_servicevertrieb',
                              'hardwar_stwar',
                              'hardwar_cloud',
                              'entwickl_objektorientiert',
                              'entwickl_salesforcemarketgcloudexpert',
                              'entwickl_cloudnativ',
                              'entwickl_sapcommercetechnologi',
                              'entwickl_sap',
                              'technolog_entwickl',
                              'websit_entwickl',
                              'architektur_entwickl',
                              'javascript_entwickl',
                              'programmier_datenbank',
                              'stwareentwickl_datenbank',
                              'erfahr_stwareentwickl',
                              'kenntnis_stwareentwickl',
                              'agil_stwareentwickl',
                              'objektorientiert_programmier',
                              'data_management',
                              'data_analytics',
                              'data_legislation',
                              'agil_devops'
                              ],
    'artificial_inelligence': [
        'deep_learning',
        'machine_learning', #!!!!!!!!!!!!!!!!!!!!!!!!!!!
        'ki_use',
        'ki_gern',
        'big_data',
        'ki_analyticsknowhow',
        'scienc_ai',
        'ki_use',
        'analytics_ki',
        'affitat_ki',
        'data_scienc',
        'artificial_telligenc',
        'ai_umgang',
        'ai_projekt',
        'datascienc_ki',
        'deep_learng',
        'nlp_data',
        'ai_iot', 'ai_umgang', 'analytics_ai', 'learng_ai',
        'learng_big', 'learng_bigdata', 'learng_comput', 'learng_deep',
        'learng_framework', 'learng_mach', 'learng_modell', 'mach_learng'

    ],
    'foreign_languages': [
        'englischkenntnis_verhlungssich', 'niederland_vorteil',
        'fremdsprach_vorteil', 'fremdsprach_sd', 'fremdsprach_vorteilhaft',
        'englischkenntnis_b1',
        'englischkenntnis_franzosischkenntnis',
        'niederland_spanisch', 'spanisch_eos', 'spanisch_exzellent', 'wunschenswert_spanisch',
        'konversationssich_englischkenntnis',
        'gut_englischkenntnis',
        'schrift_englischkenntnis',
        'englischkenntnis_wort',
        'englisch_sprachkenntnis',
        'deutsch_englischkenntnis',
        'weit_fremdsprach',
        'fremdsprach_vorteil',
        'fliessend_englischkenntnis',
        'fliessend_deutschkenntnis',
        'fliessend_englisch',
        'spanisch_exzellent',
        'fliessend_deutsch',
        'englischkenntnis_wort',
        'englischkenntnis_projektbezog',
        'englischkenntnis_eos',
        'englischkenntnis_b2',
        'deutsch_c2',
        'englisch_b2',
        'englischkenntnis_hoh',
        'englischsprach_kommunikation',
        'deutsch_englisch',
        'franzos_wunschenswert',
        'fremdsprachenkenntnis_englisch',
        'englisch_franzos',
        'franzos_niederland', 'kenntnis_niederland', 'niederland_spanisch', 'niederland_sprach', 'niederland_vorteil',
        'franzos_wunschenswert', 'fremdsprach_franzos',
        'fremdsprach_vorteil', 'fremdsprach_vorteilhaft', 'relevant_fremdsprach', 'weit_fremdsprach',
        'italien_polnisch', 'italien_spanisch', 'italien_wunschenswert', 'kenntnis_italien', 'spanisch_italien',
        'franzos_italien', 'italien_eos',
        'englisch_niederland', 'franzos_niederland', 'niederland_spanisch', 'niederland_vorteil',
        'chesisch_sprachkenntnis', 'deutsch_sprachkenntnis', 'englisch_sprachkenntnis',
        'sprachkenntnis_fliessend', 'sprachkenntnis_franzos',

    ],
    'local_language':[],
    'project_management': [
        'hervorrag_projektmanagementerfahr',
        'startupmentalitat_agil',
        'erfahr_fuhrung', 'fuhrung_arbeit', 'fuhrung_data',
        'jira_confluenc',
        'agil_projekt',
        'abteilungsubergreif_projekt',
        'projekt_verhlungssich', 'projekt_weiterzubildenntteamgeist',
        'erfahr_agil',
        'agil_projektmanagement',
        'agil_chang',
        'agil_projektumfeld', 'agil_projektvorgehensmodell', 'agil_unternehm',
        'agil_vorgehensweis', 'agil_mdset', 'agil_werkzeug', 'agil_coach', 'agil_method', 'agil_projekt',
        'traditionell_agil', 'agil_arbeit', 'agil_devops', 'agil_stwareentwickl', 'agil_unternehmensberat',
        'kenntnis_leantechn',
        'teamfah_agil',
        'agil_projektarbeit',
        'project_management',
        'zertifizier_project',
        'fuhrung_projektleitungserfahr',
        'projekt_kenakquis',
        'ubernehm_projekt',
        'projekt_manag',
        'projekt_ternational',
        'projekt_terdisziplar '
        'projekt_scrum ',
        'projekt_manag',
        'projekt_produktmanagement',
        'projekt_prozessmanagement',
        'agil_transformation',
        'agil_organisationsentwicklungsmethod',
        'agil_kontext',
        'agil_arbeitsmethod',
        'agil_coach',
        'agil_mdset',
        'agil_vorgehensweis',
        'agil_werkzeug',
        'agil_projektteam',
        'agil_projektmethod',
        'agil_umfeld',
        'agil_projektarbeit',
        'scrum_mast',
        'method_scrum',
        'agil_method',
        'agil_modell',
        'org_scrum',
        'scrum_allianc', 'scrum_ipma', 'scrum_kanban', 'scrum_mast'
    ],
    'experience': [
        'verfug_langjahr', 'langjahr_erfahr',
        'berufserfahr_mdest',
        'bereit_berufserfahr',
        'bereit_erfahr',
        'funf_jahr',
        'jahr_berufserfahr',
        'relevant_berufserfahr',
        'mehrjahr_berufserfahr',
        'bisher_berufserfahr',
        'eschlag_berufserfahr',
        'relevant_berufserfahr',
        'langjahr_erfahr',
        'mehrjahr_praxis',
        'mehrjahr_erfahr',
        'jahr_praxis',
        'jahr_erfahr',
        'eig_jahr',
        'jahr_berufserfahr',
        'drei_jahr',
        'jahr_erfahr',
        'mehrjahr_berufserfahr',
        'eschlag_berufserfahr',
        'bereit_berufserfahr',
        'relevant_berufserfahr'
    ]
}

get_top_word_combinartions('ms', list_bi_prob)
get_top_word_combinartions('mba', list_bi_prob)
get_top_word_combinartions('learn', list_bi_prob)
get_top_word_combinartions('phd', list_bi_prob) #!
get_top_word_combinartions('empath', list_bi_prob)


dic_requirements['denkweis_kreativ'] = dic_requirements['denkweis_kreativ'] + get_top_word_combinartions('kreativ',
                                                                                                         list_bi_prob)
dic_requirements['denkweis_kreativ'] = dic_requirements['denkweis_kreativ'] + get_top_word_combinartions('kreativitat',
                                                                                                         list_bi_prob)
dic_requirements['denkweis_analytisch'] = dic_requirements['denkweis_analytisch'] + get_top_word_combinartions('zahlenverstandnis',
                                                                                                         list_bi_prob)
dic_requirements['denkweis_analytisch'] = dic_requirements['denkweis_analytisch'] + get_top_word_combinartions('strukturiert',
                                                                                                         list_bi_prob)
dic_requirements['denkweis_analytisch'] = dic_requirements['denkweis_analytisch'] + get_top_word_combinartions('analysefah',
                                                                                                         list_bi_prob)
dic_requirements['denkweis_analytisch'] = dic_requirements['denkweis_analytisch'] + get_top_word_combinartions('analyt',
                                                                                                         list_bi_prob)
dic_requirements['denkweis_analytisch'] = dic_requirements['denkweis_analytisch'] + get_top_word_combinartions('zahlenaffitat',
                                                                                                         list_bi_prob)

##################################
# PROJECT MGMT

dic_requirements['project_management'] = dic_requirements['project_management'] + get_top_word_combinartions('kanban',
                                                                                                             list_bi_prob)
dic_requirements['project_management'] = dic_requirements['project_management'] + get_top_word_combinartions('scrum',
                                                                                                             list_bi_prob)
dic_requirements['project_management'] = dic_requirements['project_management'] + get_top_word_combinartions('oracle',
                                                                                                             list_bi_prob)
dic_requirements['project_management'] = dic_requirements['project_management'] + get_top_word_combinartions('agil',
                                                                                                             list_bi_prob)
dic_requirements['project_management'] = dic_requirements['project_management'] + get_top_word_combinartions('jira',
                                                                                                             list_bi_prob)
dic_requirements['project_management'] = dic_requirements['project_management'] + get_top_word_combinartions('fuhrung',
                                                                                                             list_bi_prob)


##################################
#

dic_requirements['foreign_languages'] = dic_requirements['foreign_languages'] + get_top_word_combinartions('franzos',
                                                                                           list_bi_prob)
dic_requirements['foreign_languages'] = dic_requirements['foreign_languages'] + get_top_word_combinartions('fremdsprach',
                                                                                           list_bi_prob)
dic_requirements['foreign_languages'] = dic_requirements['foreign_languages'] + get_top_word_combinartions('spanisch',
                                                                                           list_bi_prob)
dic_requirements['foreign_languages'] = dic_requirements['foreign_languages'] + get_top_word_combinartions('polnisch',
                                                                                           list_bi_prob)
dic_requirements['foreign_languages'] = dic_requirements['foreign_languages'] + get_top_word_combinartions('chesisch',
                                                                                           list_bi_prob)



##################################
# EDUCATION



dic_requirements['educ_promotion'] = dic_requirements['educ_promotion'] + get_top_word_combinartions('promotion',
                                                                                               list_bi_prob)

dic_requirements['educ_promotion'] = dic_requirements['educ_promotion'] + get_top_word_combinartions('dilpom',
                                                                                               list_bi_prob)


dic_requirements['educ_promotion'] = dic_requirements['educ_promotion'] + get_top_word_combinartions('diplomfanzwirt',
                                                                                               list_bi_prob)

dic_requirements['educ_master'] = dic_requirements['educ_master'] + get_top_word_combinartions('master',
                                                                                               list_bi_prob)

dic_requirements['educ_master'] = dic_requirements['educ_master'] + get_top_word_combinartions('mba',
                                                                                               list_bi_prob)
dic_requirements['educ_master'] = dic_requirements['educ_master'] + get_top_word_combinartions('mast',
                                                                                               list_bi_prob)
dic_requirements['educ_master'] = dic_requirements['educ_master'] + get_top_word_combinartions('masterstudium',
                                                                                               list_bi_prob)
dic_requirements['educ_bachelor'] = dic_requirements['educ_bachelor'] + get_top_word_combinartions('bachelor',
                                                                                                   list_bi_prob)
dic_requirements['educ_bachelor'] = dic_requirements['educ_bachelor'] + get_top_word_combinartions('bachelorstudium',
                                                                                                   list_bi_prob)
dic_requirements['educ_ausbildung'] = dic_requirements['educ_ausbildung'] + get_top_word_combinartions('ausbild',
                                                                                                   list_bi_prob)


##################################
# COMMUNICATION


dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'kommunikativ', list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'schreibtalent', list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'kommunikationsstark',    list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'kontaktfreud',    list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'kommunikationsgeschick',    list_bi_prob)

dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'kommunikationstalent',    list_bi_prob)

dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'sozialkompetenz',     list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'empath',    list_bi_prob)

dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'prasentationsstark',     list_bi_prob)

dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'kommunikationsfah',    list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'moderationsstark',     list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'sozialkompetenz',    list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'teamfah',    list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'teamplay',    list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'teamwork',    list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'teamorientier',    list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'zusammenzuarbeit',    list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'kooperationsstark',    list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'ausdrucksstark',    list_bi_prob)
dic_requirements['soft_skills_communication_team'] = dic_requirements[
                                                         'soft_skills_communication_team'] + get_top_word_combinartions(
    'kenorientier',    list_bi_prob)

##################################
# TRAVEL


dic_requirements['travel'] = dic_requirements['travel'] + get_top_word_combinartions('reisebereitschaft',
                                                                                               list_bi_prob)
dic_requirements['travel'] = dic_requirements['travel'] + get_top_word_combinartions('reisetat',
                                                                                               list_bi_prob)


##################################
# PROGRAMMING LANGUAGE


dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('devops',
                                                                                               list_bi_prob)
dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('stwareentwicklungsmethod',
                                                                                               list_bi_prob)

dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('anwendungsentwickl',
                                                                                               list_bi_prob)

dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('java',
                                                                                               list_bi_prob)

dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('php',
                                                                                               list_bi_prob)

dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('html',
                                                                                               list_bi_prob)


dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('sql',
                                                                                               list_bi_prob)

dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('python',
                                                                                               list_bi_prob)

dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('datenbank',
                                                                                               list_bi_prob)

dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('html5',
                                                                                               list_bi_prob)

dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('programmier',
                                                                                               list_bi_prob)

dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('cloud',
                                                                                               list_bi_prob)

dic_requirements['programming_languages'] = dic_requirements['programming_languages'] + get_top_word_combinartions('html',
                                                                                               list_bi_prob)
##########################3


dic_requirements['pc_specialized'] = dic_requirements['pc_specialized'] + get_top_word_combinartions('googl', list_bi_prob)

dic_requirements['pc_specialized'] = dic_requirements['pc_specialized'] + get_top_word_combinartions('oracl', list_bi_prob)

dic_requirements['pc_specialized'] = dic_requirements['pc_specialized'] + get_top_word_combinartions('erpsystem', list_bi_prob)

dic_requirements['pc_specialized'] = dic_requirements['pc_specialized'] + get_top_word_combinartions('sap', list_bi_prob)


get_top_word_combinartions('hardwar', list_bi_prob)

dic_requirements['pc_basic'] = dic_requirements['pc_basic'] + get_top_word_combinartions('ms', list_bi_prob)


##################################
# LOCAL LANGUAGE


dic_requirements['local_language'] = dic_requirements['local_language'] + get_top_word_combinartions('deutsch',
                                                                                               list_bi_prob)

dic_requirements['local_language'] = dic_requirements['local_language'] + get_top_word_combinartions('deutschkenntnis',
                                                                                               list_bi_prob)



##################################
# RESILIENCE

dic_requirements['soft_skills_resilience'] = dic_requirements['soft_skills_resilience'] + get_top_word_combinartions(
    'leistungsbereitschaft', list_bi_prob
)
dic_requirements['soft_skills_resilience'] = dic_requirements['soft_skills_resilience'] + get_top_word_combinartions(
    'hsonmentalitat', list_bi_prob
)
dic_requirements['soft_skills_resilience'] = dic_requirements['soft_skills_resilience'] + get_top_word_combinartions(
    'selbststand', list_bi_prob
)
dic_requirements['soft_skills_resilience'] = dic_requirements['soft_skills_resilience'] + get_top_word_combinartions(
    'esatzbereitschaft', list_bi_prob
)
dic_requirements['soft_skills_resilience'] = dic_requirements['soft_skills_resilience'] + get_top_word_combinartions(
    'flexibilitat', list_bi_prob
)
dic_requirements['soft_skills_resilience'] = dic_requirements['soft_skills_resilience'] + get_top_word_combinartions(
    'hson', list_bi_prob
)
dic_requirements['soft_skills_resilience'] = dic_requirements['soft_skills_resilience'] + get_top_word_combinartions(
    'belastbar', list_bi_prob
)
dic_requirements['soft_skills_resilience'] = dic_requirements['soft_skills_resilience'] + get_top_word_combinartions(
    'leistungsorientiert', list_bi_prob
)

#################################
# LEADERSHIP


dic_requirements['leadership_skills'] = dic_requirements['leadership_skills'] + get_top_word_combinartions(
    'loyalitat', list_bi_prob
)
dic_requirements['leadership_skills'] = dic_requirements['leadership_skills'] + get_top_word_combinartions(
    'organisationsstark', list_bi_prob
)
dic_requirements['leadership_skills'] = dic_requirements['leadership_skills'] + get_top_word_combinartions(
    'vertrauenswurd', list_bi_prob
)
dic_requirements['leadership_skills'] = dic_requirements['leadership_skills'] + get_top_word_combinartions(
    'tegritat', list_bi_prob
)
dic_requirements['leadership_skills'] = dic_requirements['leadership_skills'] + get_top_word_combinartions(
    'durchsetzungsvermog', list_bi_prob
)
dic_requirements['leadership_skills'] = dic_requirements['leadership_skills'] + get_top_word_combinartions(
    'uberzeugungsfah', list_bi_prob
)
dic_requirements['leadership_skills'] = dic_requirements['leadership_skills'] + get_top_word_combinartions(
    'authent', list_bi_prob
)
dic_requirements['leadership_skills'] = dic_requirements['leadership_skills'] + get_top_word_combinartions(
    'charakterstark', list_bi_prob
)
dic_requirements['leadership_skills'] = dic_requirements['leadership_skills'] + get_top_word_combinartions(
    'vertrauensvoll', list_bi_prob
)
dic_requirements['leadership_skills'] = dic_requirements['leadership_skills'] + get_top_word_combinartions(
    'organisationstalent', list_bi_prob
)
dic_requirements['leadership_skills'] = dic_requirements['leadership_skills'] + get_top_word_combinartions(
    'proaktiv', list_bi_prob
)

# reverse order in all bigrams to increase vocabulary
for theme in dic_requirements:
    for i in range(len(dic_requirements[theme])):
        word = dic_requirements[theme][i]
        list_words = re.findall(r'[^_]*', word)
        if len(list_words) > 2:
            reversed_combination = list_words[2] + '_' + list_words[0]
            dic_requirements[theme].append(reversed_combination)

nr_bigrams_total = 0
for theme in dic_requirements:
    nr_bigrams_total = len(dic_requirements[theme]) + nr_bigrams_total
    print(f"Number of token in theme {theme}: {len(dic_requirements[theme])}")

###########################################
###########################################
###########################################

# for each business field: which skills are the most common?


ad_i = 250

dic_count_themes = {}
for theme in dic_requirements:
    dic_count_themes[theme] = 0

avg_list_size = []
print('_' * 80)
for ad_i in range(0, 200):
    ad_i = ad_i + 1
    t1 = df1.iloc[ad_i, :].text_clean_short
    bigrams = []
    stem_germ = SnowballStemmer('german')
    nltk.word_tokenize(t1)
    for i in range(len(t1.split()) - 1):
        word_1 = stem_germ.stem(t1.split()[i])
        word_2 = stem_germ.stem(t1.split()[i + 1])
        bigrams.append(word_1 + '_' + word_2)

    bigrams = np.array(bigrams)
    # print(bigrams)
    list_found = []
    for theme in dic_requirements:
        for bi in bigrams:
            if bi in dic_requirements[theme]:
                # print(theme, ': ', bi)
                list_found.append(theme)
                dic_count_themes[theme] = dic_count_themes[theme] + 1
    # print(len(list_found))
    if len(list_found) < 3:
        print(bigrams)
        print(len(list_found))
    avg_list_size.append(len(list_found))

print(np.mean(avg_list_size))
len(avg_list_size)
print('Percent good: ', len(np.nonzero(avg_list_size)[0]) / len(avg_list_size))

# print(dic_count_themes)

#################################
#  how often does each category appear?
# idea: for each job ad, show nr. of terms per category or just nr of job ads with at least one mention of theme?


dic_job_bigrams = {}

avg_list_size = []
print('_' * 80)
for ad_i in range(0, 50):
    ad_i = ad_i + 1
    t1 = df1.iloc[ad_i, :].text_clean_short
    bigrams = []
    stem_germ = SnowballStemmer('german')
    nltk.word_tokenize(t1)
    dic_job_bigrams[ad_i] = {}
    for i in range(len(t1.split()) - 1):
        word_1 = stem_germ.stem(t1.split()[i])
        word_2 = stem_germ.stem(t1.split()[i + 1])
        bigrams.append(word_1 + '_' + word_2)
    bigrams = np.array(bigrams)
    print(bigrams)
    list_found = []
    for theme in dic_requirements:
        dic_job_bigrams[ad_i][theme] = 0
        for bi in bigrams:
            if bi in dic_requirements[theme]:
                # print(theme, ': ', bi)
                dic_job_bigrams[ad_i][theme] = dic_job_bigrams[ad_i][theme] + 1

dic_job_bigrams
df_themes_jobs = pd.DataFrame().from_dict(dic_job_bigrams, orient='index')

###########################################################3

# Q: typical skills for each field


seniority = "Unknown"
field_name = 'market*'

df1 = df_skillset.copy()
# df1 = df1.iloc[:18000, ]

for seniority in ["Unknown", "JR", "SR", "PR"]:

    print('_' * 50, 'seniority level:', seniority)
    # define field names

    list_field_names = [
        field_names_finance,
        field_names_mkt,
        field_names_accounting,
        field_names_ds,
        field_names_consulting,
        field_names_sales,
        field_names_supply,
        field_names_pr,
        field_names_hr,
        field_names_projmag]

    list_fields = [
        'finance',
        'mkt',
        'accounting',
        'ds',
        'consult',
        'sales',
        'supply',
        'pr',
        'hr',
        'projmag'
    ]

    df_comparison_fields = pd.DataFrame(columns=list(dic_requirements))



    for field_name in list_field_names:
        print('_' * 20, field_name, '_' * 40)

        local_df = df_skillset.copy()
        marketing_jobs = np.where(local_df['title'].apply(lambda x: re.findall(field_name, x.lower())),
                                  local_df.index,
                                  np.nan)

        local_df['title'].apply(lambda x: re.findall(field_name, x.lower()))
        local_df.reset_index()
        marketing_jobs
        pd.notna(marketing_jobs).sum()
        ma_index = marketing_jobs[pd.notna(marketing_jobs)]

        ma_index = ma_index.astype(int)
        df_subset = local_df.iloc[ma_index, :]
        df_subset = df_subset[df_subset['seniority'] == seniority]
        print(f"Size of subset for {seniority}: {df_subset.shape[0]} ")

        df_subset = df_subset.reset_index().drop('index',axis=1)
        dic_job_bigrams = {}

        avg_list_size = []
        print('_' * 80)
        for ad_i in range(0, df_subset.shape[0] - 1):
            ad_i = ad_i + 1
            t1 = df_subset.iloc[ad_i, :].text_clean_short
            bigrams = []
            stem_germ = SnowballStemmer('german')
            nltk.word_tokenize(t1)
            dic_job_bigrams[ad_i] = {}
            for i in range(len(t1.split()) - 1):
                word_1 = stem_germ.stem(t1.split()[i])
                word_2 = stem_germ.stem(t1.split()[i + 1])
                bigrams.append(word_1 + '_' + word_2)
            bigrams = np.array(bigrams)
            # print(bigrams)
            list_found = []
            for theme in dic_requirements:
               # theme= 'reisebereit'
                dic_job_bigrams[ad_i][theme] = 0
                for bi in bigrams:
                    if bi in dic_requirements[theme]:
                        # print(theme, ': ', bi)
                        dic_job_bigrams[ad_i][theme] = dic_job_bigrams[ad_i][theme] + 1

        # dic_job_bigrams[7]
        df_themes_jobs = pd.DataFrame().from_dict(dic_job_bigrams, orient='index')

        dic_summary = {}
        for col in df_themes_jobs.columns:
            print(col, np.round(len(np.nonzero(list(df_themes_jobs[col]))[0]) / df_themes_jobs.shape[0], 2))
            dic_summary[col] = np.round(len(np.nonzero(list(df_themes_jobs[col]))[0]) / df_themes_jobs.shape[0], 2)

        df_comparison_fields = pd.concat([df_comparison_fields, pd.Series(dic_summary).to_frame().transpose()])
        print(df_comparison_fields.shape[0])


    df_comparison_fields.index = list_fields

    df_comparison_fields.to_csv(
        f'./data_processing/topic modelling/bigrams_exported_datacomparison_fields_{seniority}.csv')



########################
# biggest changes in requirements within field

comp_jr = pd.read_csv('./data_processing/topic modelling/bigrams_exported_datacomparison_fields_JR.csv')
comp_sr = pd.read_csv('./data_processing/topic modelling/bigrams_exported_datacomparison_fields_SR.csv')
comp_un = pd.read_csv('./data_processing/topic modelling/bigrams_exported_datacomparison_fields_Unknown.csv')

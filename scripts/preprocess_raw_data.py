# This script is intended to make sure different columns in the dataset are valid, e.g. date


import numpy as np
import pandas as pd
import pandas as pd
import datetime
import os
import re
from bs4 import BeautifulSoup

dic_files = {}
dic_files['company'] = []
dic_files['location'] = []
dic_files['jobTitle'] = []
dic_files['timePosted'] = []
dic_files['estimatedDatePosted'] = []
dic_files['dateCollected'] = []
dic_files['content'] = []



for folder in os.listdir('./scripts/data_collecting'):
    if f'raw_data_{folder}' in os.listdir(f'./scripts/data_collecting/{folder}/'):
        files = os.listdir(f'./scripts/data_collecting/{folder}/raw_data_{folder}')
        print('_'*10,folder, len(files),'_'*70)
        for file_name in files:
            # print(file_name)
            content = open(f'./scripts/data_collecting/{folder}/raw_data_{folder}/{file_name}', 'r', encoding='utf-8')
            job_content = content.read()
            file_detailes = re.findall(r'.*?_', file_name)
            if len(file_detailes) <4:
                continue
            try:
                datetime.datetime.strptime(file_detailes[4][:-1], '%Y-%m-%d')
                date_collected = re.findall(r'_\d{1,}.\d{1,}.\d{1,}[.]txt', file_name)[0][1:-4]
                dic_files['company'].append(file_detailes[0][:-1])
                dic_files['location'].append(file_detailes[1][:-1])
                dic_files['jobTitle'].append(file_detailes[2][:-1])
                dic_files['timePosted'].append(file_detailes[3][:-1])
                dic_files['estimatedDatePosted'].append(file_detailes[4][:-1])
                dic_files['dateCollected'].append(date_collected)
                dic_files['content'].append(BeautifulSoup(job_content, 'lxml').text)
            except:
                print('invalid date format')
            content.close()

print(f"Dictionary has a total of {len(dic_files['company'])} jobs.")



df_jobs = pd.DataFrame().from_dict(dic_files, orient='columns')


import datetime
import numpy as np


# df_jobs['estimatedDatePosted'] = datetime(df_jobs['estimatedDatePosted'])



df_jobs['estimatedDatePosted'] = pd.to_datetime(df_jobs['estimatedDatePosted'], format='%Y-%m-%d')


df_jobs.to_csv('./scripts/data_preprocessing/preprocessed_jobs.csv')



# to here


################################################################################



for year in range(2006, 2022,1):
    jobs_per_year = np.sum(df_jobs['estimatedDatePosted'].dt.year == int(year))
    unique_companies = len(df_jobs[df_jobs['estimatedDatePosted'].dt.year == int(year)].company.unique())
    print(f"Total jobs in {year}: {jobs_per_year} from {unique_companies} unique companies")



x = df_jobs[df_jobs['estimatedDatePosted'].dt.year < 2010]['jobTitle']


x = df_jobs[df_jobs['estimatedDatePosted'].dt.year < 2010].reset_index()
print(x.iloc[2].jobTitle)



res = df_jobs['jobTitle'].apply(lambda x: re.findall(r'.{0,}Tax.{0,}',x))
res.sum()


a = np.where(df_jobs['jobTitle'].apply(lambda x: len(re.findall(r'.{0,}Tax.{0,}',x)))>0,True,False)

np.unique(a)
np.where(a==1)
filtered = df_jobs[a]
filt_sort = filtered[filtered['estimatedDatePosted'].dt.year == 2021].reset_index()
aa = filt_sort.iloc[1].content
ab = x.iloc[2].content
print(aa[40:])
print(ab[40:])

re.findall('qualifi.*',aa)[0]
re.findall('Profil.*',ab)[0]

ii = 0
while True:
    print(aa[ii:ii+200],'\n')
    ii = ii+200
    if ii >= len(aa):
        break


ii = 0
while True:
    print(ab[ii:ii+200],'\n')
    ii = ii+200
    if ii >= len(ab):
        break


res_ar = res.array
res_ar
np.array(res_ar)



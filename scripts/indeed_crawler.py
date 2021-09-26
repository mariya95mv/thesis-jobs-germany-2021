

import pyttsx3
from datetime import datetime
import numpy as np
from selenium import webdriver
import time
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import re
import sys
from selenium.webdriver.firefox.options import Options

sys.path.append("C:/Users/Maria/PycharmProjects/Thesis")
sys.path.append("C:/Users/Maria/PycharmProjects/Thesis/scripts")
from scripts.data_collecting.global_data_collection_functions.global_functions import remove_signs, save_job_ad, remove_HTML_tokens, clean_job_description


folder_to_save = "raw_data_indeed"

path_gecko="C:/Users/Maria/geckodriver-v0.28.0-win64/geckodriver.exe"
sys.path.append(path_gecko)
sys.path.append("C:/Users/Maria/PycharmProjects/lang_poc")
sys.path.append("C:/Users/Maria/")
sys.path.append("C:/Program Files (x86)/Mozilla Firefox")
sys.path.append("C:/Users/Maria/geckodriver-v0.28.0-win64")
bi = FirefoxBinary("C:/Program Files (x86)/Mozilla Firefox/firefox.exe")
opt = Options()
opt.binary = bi

engine = pyttsx3.init()

first_job = None



class Description():
    def __init__(self, text):
        self.text = None
        self.set_text(text)
    def set_text(self, text):
        self.text = text

job_categories = ['junior+marketing', 'junior+sales','junior+analyst', 'junior+controlling',
                  'junior+auditing','junior+finance', 'junior+hr', 'junior+data+science', 'junior+consulting',
                  'junior+pr']


np.random.shuffle(job_categories)

for job_idx, category in enumerate(job_categories):
    next_category = False
    print('_'*120)
    print(' '*20, f'Current category for indeed query: {category}, {job_idx}/{len(job_categories)}')
    print("_"*120)
    page = 0

    message = f'Starting category {category}'
    engine.say(message)
    engine.runAndWait()
    print(message)
    print('--'*20)

    while True:
        print("*********************************************************************************")
        print(f"Page {page}")

        driver = webdriver.Firefox(firefox_options=opt,
                                   executable_path="C:/Users/Maria/geckodriver-v0.28.0-win64/geckodriver.exe")
        driver_href = webdriver.Firefox(firefox_options=opt,
                                        executable_path="C:/Users/Maria/geckodriver-v0.28.0-win64/geckodriver.exe")

        url = f"https://de.indeed.com/Jobs?q={category}&l=Deutschland&start={page}&sort=date"
        driver.get(url)
        time.sleep(6)
        jobs = driver.find_elements_by_class_name('jobTitle')
        time_in_loop = 0
        to_cont = 'y'
        while len(jobs) < 1:
            engine.say("Page error detected. Go to indeed.")
            engine.runAndWait()
            print("_________________________ PAGE  ERROR  ______________________________")
            continue_scrape = input("No jobs found. press 'r' to refresh, 'n' to go to next page:")
            if continue_scrape == 'r':
                driver.refresh()
                time.sleep(5)
                jobs = driver.find_elements_by_class_name('jobTitle')
            elif continue_scrape == 'n':
                to_cont = 'n'
                break

        if to_cont == 'n':
            driver.close()
            driver_href.close()
            time.sleep(5)
            break

        if next_category == True:
            next_category = False
            break

        print('Loading jobs...')
        jobs_list = []
        for job in jobs:
            jobs_list.append(job.text)
            time.sleep(3)

        if first_job == jobs_list[0]:
            break
        else:
            first_job = jobs_list[0]
        print("Jobs loaded.")
        job_meta_location = driver.find_elements_by_class_name('heading6')
        # driver.find_elements_by_class_name('companyLocation')[10].text
        # links = driver.find_elements_by_css_selector(".mosaic-zone-jobcards [href]")

        # x2 = driver.find_elements_by_id('mosaic-zone-jobcards')[0]
        # x2.find_elements_by_tag_name('a')[25].get_attribute('href')
        # len(x2.find_elements_by_tag_name('a'))
        # x2.find_elements_by_tag_name('a')

        # x1 = driver.find_elements_by_id('mosaic-provider-jobcards')[0]
        # x1.find_elements_by_tag_name('a')[0].get_attribute('href')
        # len(x1.find_elements_by_tag_name('a'))

        x1 = driver.find_elements_by_id('mosaic-provider-jobcards')[0]
        time.sleep(2)
        links = x1.find_elements_by_class_name('tapItem')
        # links[3].get_attribute('href')


        batch_counter = 0
        for i, job_raw in enumerate(jobs):
            if next_category == True:
                break
            time.sleep(3)
            job_title = remove_signs(jobs_list[i])
            link = links[i].get_attribute('href')
            link_to_ad_page = link
            location_raw = driver.find_elements_by_class_name('companyLocation')[i].text
            if len(location_raw) <1:
                print('Error: bad job description. Skipping to next job.')
                continue
            # location_list = re.findall(r'\n.*', location_raw)
            # if len(re.findall(r'.*\n', location_raw))<1:
            #     continue

            job_company = driver.find_elements_by_class_name('companyName')[i].text
            job_company = remove_signs(job_company)

            location_list = driver.find_elements_by_class_name('companyLocation')[i].text
            job_location = remove_signs(location_list)
            # if len(location_list) == 1:
            #     job_location = remove_signs(location_list[0][1:])
            # else:
            #     job_location = "Unknown"

            driver_href.get(link_to_ad_page)
            time.sleep(6)
            load_successful = False

            try:
                job_full_description = driver_href.find_element_by_id('jobDescriptionText')
            except:
                print("No job description found. Going to next job.")
                continue

            if len(job_full_description.text) > 0:
                job_full_description =  job_full_description.text
            else:
                continue

            job_description = []
            job_description.append(Description(text=job_full_description))

            job_meta = driver_href.find_elements_by_css_selector(".jobsearch-JobTab-content")
            time.sleep(2)
            if len(job_meta) ==0:
                print("Error: job meta data format wrong. Going to next job.")
                continue
            else:
                job_meta  = job_meta[0].text
            possible_time = re.findall(r'.*\n', job_meta)
            current_date = datetime.now().strftime('%Y-%m-%d')
            job_time_posted= None
            job_approx_date_posted = None

            if len(re.findall(r'\w', job_meta)) > 0:
                if len(re.findall(r'heute', job_meta.lower())) > 0:
                    job_time_posted = 'heute'
                    job_days_online =0
                elif len(re.findall(r'gerade geschaltet', job_meta.lower())) > 0:
                    job_time_posted = 'gerade geschaltet'
                    job_days_online =0
                elif len(re.findall(r'gestern', job_meta.lower())) > 0:
                    job_time_posted = 'gestern'
                    job_days_online = 1
                elif len(re.findall(r'\d{1,} minut', job_meta.lower()))> 0:
                    job_time_posted = 'heute'
                    job_days_online = 0
                elif len(re.findall(r'\d{1,} tag*', job_meta.lower()))> 0:
                    job_time_posted = 'gestern'
                    job_days_online = 1
                elif len(re.findall(r'\d{1,} day', job_meta.lower()))> 0:
                    job_time_posted = 'gestern'
                    job_days_online = 1
            if job_days_online > 0:
                print('Going to next category.')
                next_category = True
                continue


            try:

                from datetime import timedelta

                now = datetime.now()
                td_days_online = timedelta(job_days_online)

                month = (now - td_days_online).month
                day = (now - td_days_online).day
                year = (now - td_days_online).year

                job_approx_date_posted = datetime(day=day, month=month, year=year).date().strftime('%Y-%m-%d')

            except:
                if job_time_posted is None:
                    continue

                elif str(job_time_posted).lower() == 'heute':
                    job_days_online=0
                    job_approx_date_posted = current_date
                elif str(job_time_posted).lower() == 'gerade geschaltet':
                    job_days_online=0
                    job_approx_date_posted = current_date
                elif str(job_time_posted).lower() == 'gestern':
                    print("__________________________________________________")
                    print(f"This job ad is older than 24 hours. Continueing to next job.")

                    message = 'going to next category.'
                    print(message)
                    engine.runAndWait(message)
                    next_category = True
                    continue

            if re.findall(r'\d{1,} tag*',str(job_time_posted).lower()).__len__() >0:
                print("__________________________________________________")
                print(f"This job ad is older than 24 hours. Continueing to next job.")
                message = 'going to next category.'
                print(message)
                engine.runAndWait(message)
                next_category = True
                break


            if job_time_posted is None:
                message = f'This job is old ({job_meta}). Moving to next category: '
                engine.say(message)
                engine.runAndWait()
                print('_________________________________')
                print(message)
                next_category = True
                break

            file_name = f"{job_company}_{job_location}_{job_title}_{job_time_posted}_{job_approx_date_posted}_{current_date}"

            if len(file_name) > 180:
                if len(job_location) > 30:
                    print("WARNING: File name too long to save!")
                    job_location = job_location[:10]
                    file_name = f"{job_company}_{job_location}_{job_title}_{job_time_posted}_{job_approx_date_posted}_{current_date}"
                    if len(file_name) < 200:
                        batch_counter = save_job_ad(folder_to_save, file_name, job_description, batch_counter, page,
                                                job_time_posted, job_title, link)
                    else:
                        file_name = f"{job_company[:10]}_{job_location[:5]}_{job_title}_{job_time_posted}_{job_approx_date_posted}_{current_date}"
                        batch_counter = save_job_ad(folder_to_save, file_name, job_description, batch_counter, page,
                                                    job_time_posted, job_title, link)


            else:
                batch_counter = save_job_ad(folder_to_save, file_name, job_description, batch_counter, page,
                                            job_time_posted, job_title, link)

            job_days_online = None

        if next_category == True:
            time.sleep(10)
            next_category=False
            break
        else:
            page = page + 10
            driver.close()
            driver_href.close()
            random_sleep = 61
            print(f"Sleeping for {random_sleep } seconds")
            time.sleep(random_sleep)
            # if page % 50 == 0:
            #     time.sleep(15)
            #     page = page + 20
            if page >60:
                message = 'Reached page 300. Should I switch to next category? '
                print('______________________')
                print(message)
                engine.runAndWait(message)
                to_cont = input('Press "n" for next category:\n>>>>>> ')
                if to_cont == 'n':
                    next_category = True
                    break


    message= f'Starting a new category.'
    engine.say(message)
    engine.runAndWait()
    print('______________________')
    print(message)
    driver.close()
    driver_href.close()



message= 'Program finished. Press any button to close.'
engine.say(message)
engine.runAndWait()
input('>>>>> Press anything to close')


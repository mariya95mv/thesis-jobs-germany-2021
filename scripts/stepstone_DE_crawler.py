

from datetime import timedelta
from datetime import datetime
import numpy as np
from selenium import webdriver
import time
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import re
import sys
from selenium.webdriver.firefox.options import Options
import pyttsx3

sys.path.append("C:/Users/Maria/PycharmProjects/Thesis")
sys.path.append("C:/Users/Maria/PycharmProjects/Thesis/scripts")
from scripts.data_collecting.global_data_collection_functions.global_functions import remove_signs, save_job_ad, check_jobs_length

folder_to_save = "raw_data_stepstone"

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


print("____________________________________")
input_raw = input("Enter starting page (min 1, increment 25). Enter 'y' to start with default 0: ")
if input_raw == 'y':
    page = 0
else:
    page = int(input_raw)

# page 0, 25, 50 ...

while True:
    print("********************************************")
    print(f"Page {page}")

    url = f"https://www.stepstone.de/5/job-search-simple.html?newsearch=1&freetext_exact=false&freetext_all_words=false&fu=1005001&fu=1005003&fu=1008000&fu=1020000&fu=1009000&fu=2004000&fu=2010000&fu=2003000&fu=2006001&fu=2002000&fu=2009000&fu=2006004&fu=2006003&fu=2005000&fu=2001000&fu=2008000&fu=6013000&fu=6009000&fu=6004000&fu=6016000&fu=6001000&fu=6005000&fu=6012000&fu=6015000&fu=6008000&fu=6003000&fu=6014000&fu=6006000&fu=6011002&fu=6010000&fu=6011001&fu=4002003&fu=4002002&fu=4002001&fu=4001003&fu=4008002&fu=4001001&fu=4004001&fu=4001002&fu=4002004&fu=4009000&fu=4005000&fu=4010000&fu=4002005&fu=4011000&fu=4006000&fu=4003000&fu=4008001&fu=3001001&fu=3001002&fu=3004001&fu=3001005&fu=3002001&fu=3002004&fu=3004002&fu=3002002&fu=3001003&fu=3006000&fu=3001004&fu=3002006&fu=3002003&fu=3003002&fu=3005001&fu=8012000&fu=8013000&fu=8002000&fu=8008000&fu=8009000&fu=8014000&fu=8007000&fu=8015000&fu=8006000&fu=8011000&fu=8010000&fu=8005001&fu=5003000&fu=5001000&fu=5007002&fu=5004001&fu=5002000&fu=5004002&fu=5006000&fu=5005000&fu=5007001&fu=11007000&fu=11008000&fu=11005000&fu=11003000&fu=11009000&fu=11001000&fu=11010000&fu=11006000&re=353&re=50000&re=50001&re=50002&re=50003&re=50004&re=50005&re=50006&re=50007&re=50008&re=50009&re=50011&re=50021&re=50012&re=50013&re=50014&re=50015&re=50016&re=50017&re=50018&re=50019&re=50020&re=50022&re=50032&re=50023&re=50024&re=50025&re=50026&re=50027&re=50028&re=50029&re=50030&re=50031&re=50033&re=50043&re=50034&re=50035&re=50036&re=50037&re=50038&re=50039&re=50040&re=50041&re=50042&re=50044&re=50054&re=50045&re=50046&re=50047&re=50048&re=50049&re=50050&re=50051&re=50052&re=50053&re=50055&re=50065&re=50056&re=50057&re=50058&re=50059&re=50060&re=50061&re=50062&re=50063&re=50064&re=50066&re=50076&re=50067&re=50068&re=50069&re=50070&re=50071&re=50072&re=50073&re=50074&re=50075&re=50077&re=50087&re=50078&re=50079&re=50080&re=50081&re=50082&re=50083&re=50084&re=50085&re=50086&re=50088&re=50098&re=50089&re=50090&re=50091&re=50092&re=50093&re=50094&re=50095&re=50096&re=50097&re=50099&re=50109&re=50100&re=50101&re=50102&re=50103&re=50104&re=50105&re=50106&re=50107&re=50108&suid=080cff70-678f-4327-a45f-cf30ffcfb929&of={page}"

    driver = webdriver.Firefox(firefox_options=opt,
                               executable_path="C:/Users/Maria/geckodriver-v0.28.0-win64/geckodriver.exe")
    driver.get(url)
    time.sleep(5)

    # jobs = driver.find_elements_by_css_selector('div[class="sc-fzoVTD Atfvz"]')

    jobs =  driver.find_elements_by_tag_name('article')
    # articles[0].find_elements_by_tag_name('a[target]')[0].get_attribute('href')

    time.sleep(2)
    # links = driver.find_elements_by_css_selector('.sc-fzqNqU izlUaN [href]')
    # links = driver.find_elements_by_css_selector('a[class="sc-fzoVTD Atfvz"]')

    # works
    # l1 = jobs[0].find_elements_by_tag_name('a[target]')
    # l1[0].get_attribute('href')

    print(f'Length of jobs is: ', len(jobs))
    if len(jobs) < 2:
        message = 'Error: few jobs found. Look the script for step stone.'
        print('____________\n', message)
        engine.say(message)
        engine.runAndWait()
        to_continue = input('press any button to continue.... ')
        driver.refresh()
        time.sleep(15)


    jobs_list = []
    for job in jobs:
        jobs_list.append(job.text)
        time.sleep(3)

    if first_job == jobs_list[0]:
        message = "First job identical to last job. You have reached the final page. Exiting..."
        engine.say(message)
        engine.runAndWait()
        print('___________________________')
        print(message)
        break
    else:
        first_job = jobs_list[0]

    batch_counter = 0
    for i, link in enumerate(jobs):
        job = jobs[i]
        time.sleep(2)
        link = job.find_elements_by_tag_name('a[target]')[0].get_attribute('href')
        link_to_ad_page = link
        time.sleep(2)
        location_raw = jobs[i].text
        location_list = re.findall(r'.*\n', location_raw)
        if len(location_list) <4:
            print("Length of locaton list too small, going to next:")
            print(location_list)
            continue
        job_title = remove_signs(location_list[0][:-1])
        job_company = remove_signs(location_list[1][:-1])
        job_location = remove_signs(location_list[2][:-1])
        job_time_posted = remove_signs(location_list[3][:-1])
        driver_href = webdriver.Firefox(firefox_options=opt,
                                        executable_path="C:/Users/Maria/geckodriver-v0.28.0-win64/geckodriver.exe")

        driver_href.get("https://www.google.de")
        time.sleep(4)
        driver_href.get(link_to_ad_page)
        time.sleep(12)
        job_descri_elements = driver_href.find_elements_by_css_selector('div[class="sc-dfVpRl cFyqyv"]')
        time.sleep(2)

        continue_scrape = 'y'

        while len(job_descri_elements) <1:
            engine.say("Job description error detected. Go look at the script for website step stone.")
            engine.runAndWait()
            print("_________________________ PAGE  ERROR  ______________________________")
            print(">\n>\n>")
            continue_scrape = input(">>>   No jobs found. Refresh page and press any button to continue, press 'n' to go to next page: ")
            time.sleep(5)
            if continue_scrape == 'n':
                break
            job_descri_elements = driver_href.find_elements_by_css_selector('div[class="sc-dfVpRl cFyqyv"]')

        if continue_scrape == 'n':
            continue


        current_date = datetime.now().strftime('%Y-%m-%d')
        job_approx_date_posted = None

        if len(re.findall(r'\d.*minute.*', job_time_posted)) ==1:
            job_days_online = 0
        elif len(re.findall(r'\d.*hour.*', job_time_posted)) == 1:
            job_days_online = 0
        elif len(re.findall(r'today', job_time_posted)) == 1:
            job_days_online=0
        elif len(re.findall(r'days.*', job_time_posted)) == 1:
            engine.say("All jobs for today were saved. Exiting.")
            engine.runAndWait()
            break
            # job_days_online= int(re.findall(r'\d{1,}', job_time_posted)[0])
        elif len(re.findall(r'week.*', job_time_posted)) ==1:
            engine.say("All jobs for today were saved. Exiting.")
            engine.runAndWait()
            break
            # weeks = int(re.findall(r'\d{1,}', job_time_posted)[0])
            # job_days_online = 7*weeks
        else:
            print('Timedelta invalid, going to next job ad.')
            continue

        now = datetime.now()
        td_days_online = timedelta(job_days_online)

        month = (now - td_days_online).month
        day = (now - td_days_online).day
        year = (now - td_days_online).year
        job_approx_date_posted = datetime(day=day, month=month, year=year).date().strftime('%Y-%m-%d')
        file_name = f"{job_company}_{job_location}_{job_title}_{job_time_posted}_{job_approx_date_posted}_{current_date}"
        if len(file_name) > 200:
            if len(job_location)> 30:
                locations = re.findall(r'\w.*?\s', job_location)
                for location in locations:
                    file_name = f"{job_company}_{location}_{job_title}_{job_time_posted}_{job_approx_date_posted}_{current_date}"
                    batch_counter = save_job_ad(folder_to_save, file_name, job_descri_elements, batch_counter, page,
                                            job_time_posted, job_title, link)
        else:
            batch_counter = save_job_ad(folder_to_save, file_name, job_descri_elements, batch_counter, page,
                                        job_time_posted, job_title, link_to_ad_page)

            job_days_online = None

        driver_href.close()

    page = page + 50
    driver.close()
    if page % 10 == 0:
        page = page + 25
    print("__________________________________________________")
    print(f"Sleeping for 62 seconds")
    time.sleep(62)

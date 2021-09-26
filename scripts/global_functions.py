

import pyttsx3
import  time
from datetime import datetime
import re

def x():
    print("Success")


def remove_signs(text):
    text = text.replace(':', " ")
    text = text.replace('.', " ")
    text = text.replace('...', " ")
    text = text.replace(',', " ")
    text = text.replace('(', " ")
    text = text.replace(')', " ")
    text = text.replace('/', " ")
    text = text.replace('|', " ")
    text = text.replace('\n', " ")
    text = text.replace('*', " ")
    text = text.replace('<', " ")
    text = text.replace('>', " ")
    text = text.replace('^', " ")
    text = text.replace('%', " ")
    text = text.replace('!', " ")
    text = text.replace('?', " ")
    text = text.replace(';', " ")
    text = text.replace('-', " ")
    text = text.replace('@', " ")
    text = text.replace('\\', " ")
    text = text.replace('&', "and")
    text = text.replace('"', " ")
    text = text.replace("'", " ")
    text = text.strip()
    text = re.sub(' +', ' ', text)
    return text



def save_job_ad(folder_to_save, file_name, job_descri_elements, batch_counter, page, job_time_posted, job_title, link_to_ad_page):
    f = open(
        # f'./scripts/data_collecting/{folder_to_save}/{file_name}.txt',
        f'./{folder_to_save}/{file_name}.txt',
        'w', encoding="utf-8")

    for each in job_descri_elements:
        f.write(each.text + "\n")
    f.close()
    print(f"      File {batch_counter}, page{page} saved. Time saved: {datetime.now().hour}:{datetime.now().minute} Posted: {job_time_posted}, Title: {job_title}, link: {str(link_to_ad_page)[:25]}")
    batch_counter = batch_counter + 1
    return batch_counter


def remove_HTML_tokens(text):
                text = text.replace(" ul "," ")
                text = text.replace(" div ", " ")
                text = text.replace(" li ", " ")
                text = text.replace(" br ", " ")
                text = re.sub(' +', ' ', text)
                text = text.replace(" ul ", " ")
                text = text.replace(" li ", " ")
                text = text.replace(" h2 ", " ")
                text = text.replace(" p ", " ")
                text = text.strip()
                return text

def clean_job_description(text):
    text = text.replace("\t"," ")
    text = text.replace("\n", " ")
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text


def check_jobs_length(jobs, engine, driver, jobs_class_name, css_selector = False):
    while len(jobs) < 1:
        engine.say("Page error detected. Madam, go look at the script for website indeed.")
        engine.runAndWait()
        print("_________________________ PAGE  ERROR  ______________________________")
        print(">\n>\n>")
        continue_scrape = input(">>>   No jobs found. Refresh page and press 1:")
        time.sleep(5)
        if css_selector == False:
            jobs = driver.find_elements_by_class_name(f"{jobs_class_name}")
        else:
            jobs = driver.find_elements_by_css_selector(f"{jobs_class_name}")
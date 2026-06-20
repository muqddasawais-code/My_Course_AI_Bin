from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import csv
import time

url="https://www.daraz.pk/tag/smart-kitchen-appliances/"
time.sleep(10)

cService = webdriver.ChromeService(executable_path='C:\\Users\\Awais\\Documents\\FULL-STACK-B-8-Wed-Thru-Fri\\Week4\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe')
driver = webdriver.Chrome(service=cService)

driver.get(url)

qouestList=[]
qoutesDiv = driver.find_elements(By.XPATH, "//div[contains(@class, 'Bm3ON')]")
for p in range(len(qoutesDiv) -1):
    quote = {}
    innerImg = qoutesDiv[p+1].find_element(By.TAG_NAME, "img")
    innera = qoutesDiv[p+1].find_element(By.TAG_NAME, "a")
    quote["img"] =innerImg.get_attribute('src') 
    quote["lines"] =innerImg.get_attribute('alt') 
    quote['url'] = innera.get_attribute('href')
    quote['price'] = qoutesDiv[p+1].find_element(By.XPATH, ".//span[contains(@class, 'ooOxS')]").text
    qouestList.append(quote)

filename = 'daraz_home_appliances_data.csv'
with open(filename, 'w', newline='') as f:
    w = csv.DictWriter(f,['img','lines','url','price'])
    w.writeheader()
    for quote in qouestList:
        w.writerow(quote)

driver.close()
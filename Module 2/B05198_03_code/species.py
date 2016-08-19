# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:45:57 2015

@author: dj
"""

import urllib
from bs4 import BeautifulSoup
import numpy as np

url = urllib.request.urlopen("http://interthing.org/dmls/species.html");
html = url.read()
soup = BeautifulSoup(html, "lxml")
table = soup.find("table")

headings = [th.get_text() for th in table.find("tr").find_all("th")]

datasets = []
for row in table.find_all("tr")[1:]:
	dataset = list(zip(headings, (td.get_text() for td in 	row.find_all("td"))))
	datasets.append(dataset)

nd=np.array(datasets)
features=nd[:,1:,1].astype('float')
targets=(nd[:,0,1:]).astype('str')
print(features)
print(targets)



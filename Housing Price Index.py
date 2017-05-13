
'Data Selection for analysis purpose'
#Reference - http://stackoverflow.com/questions/12065885/how-to-filter-the-dataframe-rows-of-pandas-by-within-in
#Reference - https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/
#Reference - http://stackoverflow.com/questions/21247992/grouping-pandas-dataframe-by-two-columns-or-more
#Reference - Dr. Gene Moo Lee's notes for DataScience

import pandas as pd

dataframe = pd.DataFrame.from_csv('HPI_master.csv')
segregated_df= dataframe[dataframe['level']=='MSA']
modified_HPI_master = pd.DataFrame(segregated_df.groupby([segregated_df.yr,segregated_df.place_name]).index_nsa.mean())
modified_HPI_master.to_csv('modified_HPI_master.csv')

'To merge two datasets'

import csv

merged_record=[]

with open('modified_HPI_master.csv','r') as HPI:
    hpi_reader = csv.reader(HPI, delimiter = ',')
    for record in hpi_reader:
        #print record[6]
        try:
           HPI_place_list=record[1].lower().split(',')
           HPI_city=HPI_place_list[0]
           HPI_state=HPI_place_list[1].split()[0]
           HPI_year = record[0]
        except:
            continue
        with open('report.csv','r') as rep:
            rep_reader = csv.reader(rep, delimiter = ',')
            for row in rep_reader:
                try:
                   report_place_list=row[2].lower().split()
                   report_city=report_place_list[0].rstrip(',')
                   report_state=report_place_list[1]
                   report_year=row[0]
                except:
                    continue
                if report_city  in HPI_city and report_state in HPI_state and HPI_year == report_year:
                    merged_record.append(list(record + row[2:9]))

               
                    

'Writing the merged list to a csv file'
with open('merged_dataset.csv','wb') as merged:
    writer = csv.writer(merged, delimiter=',')
    for row in merged_record:
        writer.writerow(row)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from pandas import ExcelWriter

xls_file = pd.ExcelFile('F:/Data Science/Project 3/merged_dataset.xlsx')
df = xls_file.parse('merged_dataset')
from pandas.tools.plotting import scatter_matrix
df=df.dropna()
writer1 = ExcelWriter('F:/Data Science/Project 3/ForTableau.xlsx')
df.to_excel(writer1,'Sheet1')
writer1.save()
scatter_matrix(df,figsize=(25,25))
plt.show()
print df.describe()


#df=df.dropna()
#withindexdf=cleaned_df.reset_index()
#print np.corrcoef(df.index_nsa,df.robberies)

x=df.drop(['index_nsa','Report_city name','violent_crimes'],1)
y=df['index_nsa']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

clf=LinearRegression()
model=clf.fit(x_train,y_train)

print("Regression Coefficients are: {}".format(model.coef_))
print ("Intercept value is {}".format(model.intercept_))

y_pred= model.predict(x_test)
print("R2 score {}".format(metrics.r2_score(y_test,y_pred)))
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.scatter(y_test,y_pred,s=df.index_nsa)
plt.show()
#print y_pred
newx_test=np.array([2017,391484,23,297,2651,611]).reshape(1,-1)
newy_pred=model.predict(newx_test)
print("Predicted Value of Wichita for 2017 is {}").format(newy_pred)

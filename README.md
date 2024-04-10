# <p align="center"> Fraud Detection System using Machine Learning </p>
# <p align="center">![image](https://github.com/AhamedSahil/CYBER-SECURITY-/assets/164605797/2c615903-036b-4e86-b6be-f918ef538681)
</p>

## Overview

This project aims to develop a machine learning-based system for detecting fraud in cyber security transactions. The system utilizes 
various supervised and unsupervised learning techniques to analyze transactional data and identify potentially fraudulent activities.
By employing advanced algorithms, the system can adapt to evolving fraud patterns and provide accurate predictions in real-time.

**Tools:-** Excel,Python

[Datasets Used](https://docs.google.com/spreadsheets/d/1Yp_rcOS2TbVn-wHUIsCeCzkeDP7MIPLP/edit?usp=sharing&ouid=102868121048017441192&rtpof=true&sd=true )

[Python Script (Code)](cyber_security.ipynb)

[Ppt presentation](sql_prjct.pptx)

### Features 

- Data preprocessing: Clean and prepare the transactional data for analysis.
  
- Supervised learning: Train classification models to classify transactions as fraudulent or legitimate.
  
- Model evaluation: Assess the performance of the models using relevant metrics such as precision, recall, and F1-score.


## Requirements

- Python 3

- Libraries: NumPy, pandas, Sklearn, etc.

- Jupyter Lab

## Balancing an unbalanced dataset:
```py
#So, we can do Undersampling technique to balance the datasets otherwise As you can see, this model is only predicting 0, which means it’s completely ignoring the minority class in favor of the majority class.
df_majority = sample[sample.Attack == 0]
df_minority = sample[sample.Attack == 1]
df_majority_undersample = df_majority.sample(replace = False, n = 144503, random_state = 123)#random_state it's won't shuffle if we run this multiple time
b_sample = pd.concat([df_majority_undersample, df_minority])
print(b_sample.Attack.value_counts())
b_sample.shape
```
```py
fig = plt.figure(figsize = (8,5))
b_sample.Attack.value_counts().plot(kind='bar', color= ['blue','green'], alpha = 0.9, rot=0)
plt.title('Distribution of data based on the Binary attacks of our balanced dataset')
plt.show()
```
###### Result: 

![image](https://github.com/AhamedSahil/CYBER-SECURITY-/assets/164605797/acf581f1-9322-4448-ab43-de832070a2ce)

## Model evaluation:
#### Decision Tree Algorithm
```py
ds=DecisionTreeClassifier(max_depth=3)
ds.fit(x_train,y_train)
train_pred=ds.predict(x_train)
test_pred=ds.predict(x_test)
print(accuracy_score(train_pred,y_train))
print(accuracy_score(test_pred,y_test))
```
```py
#creating list for train test accuracy
train_test = ['Train','test']
aucc = [dt_aucc,dt_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['green', 'skyblue'])
#Add Labels and title 
plt.xlabel('Decision Tree')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for train test')
#Show the plot
plt.show()
```

###### Result:

![image](https://github.com/AhamedSahil/CYBER-SECURITY-/assets/164605797/d051c597-d389-417b-b8bd-bb3c28882575)





- coverted  invoice column into primary key
```mysql
ALTER TABLE supermarket_sales.supermarket_sales
CHANGE COLUMN `Invoice ID` `Invoice ID` VARCHAR(100) NOT NULL,
ADD PRIMARY KEY (`Invoice ID`);
```

### OBJECTIVE OF THE STUDY

- To visualize how explanatory variable i.e., Branch, Customer type, Gender, Product line and Payment type affect to study variable sales.

- To check Main and Interaction effect of explanatory Variable on sales.
 
- Fitting appropriate Time Series Model and analyzing study variable sales.

### Contents

- Sales Performance Analysis

- Average Profit Analysis

- Sentiment Analysis

- Quantity of products sold in each city /branch

- Top selling product

### The questions we have apply on our dataset

- Finding the total sales with respect to product line and the top selling product 
```mysql
select product_line,sum(quantity) as total_sell from supermarket_sales group by product_line order by total_sell desc;
```
Result:

![image](https://github.com/AhamedSahil/Project-1/assets/164605797/e384bbc0-b2f7-4d59-9dd6-52f8665bf8aa)

- finding the most profitable product with respect to product_line and top profitable product
```mysql
select product_line,sum(gross_income) as total_profit from supermarket_sales group by product_line order by total_profit desc;#top profitable product is food and beverages.
```
Result:





















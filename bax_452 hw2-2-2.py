#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # To read file
import numpy as np # For multiple regression
import matplotlib.pyplot as plt # To plot
import statsmodels.formula.api as smf
from statsmodels.formula.api import logit
from sklearn.model_selection import train_test_split # To train and split
from sklearn.metrics import classification_report, accuracy_score #For confusion matrix



# ### Exercise 1

# In[20]:


###import the data
sales = pd.read_csv('/Users/edith/Desktop/BAX_452/hw2/sales.csv',)

###Build a model with interaction
model_interaction = smf.ols(formula='total_sales ~ area1_sales+area2_sales+area3_sales+area1_sales:area2_sales + area1_sales:area3_sales + area2_sales:area3_sales + area2_sales:area3_sales:area1_sales', data=sales).fit()
summary = model_interaction.summary()
print (summary.tables[1])
print(model_interaction.summary())
print("model with interaction's aic:",model_interaction.aic)
print("model with interaction's bic:",model_interaction.bic)

### Build a model without interaction
model_no_interaction = smf.ols(formula='total_sales ~ area1_sales+area2_sales+area3_sales', data=sales).fit()
summary_2 = model_no_interaction.summary()
print (summary_2.tables[1])
print(model_no_interaction.summary())
print("model without interaction's aic:",model_no_interaction.aic)
print("model without interaction's bic:",model_no_interaction.bic)


# ### Exercise 2

# In[25]:


cst = pd.read_csv('/Users/edith/Desktop/BAX_452/hw2/customer.csv',)
###Build Full model
f1 = 'Purchased ~ Age+Gender+EstimatedSalary'
logits = smf.logit(formula = f1, data = cst)
result = logits.fit()
print(result.summary())
print("model f1 aic:",result.aic)
print("model f1 bic:",result.bic)

### Build trim models
f2 = 'Purchased ~ Age+Gender'
logits2 = smf.logit(formula = f2, data = cst)
result2 = logits2.fit()
print(result2.summary())
print("model f2 aic:",result2.aic)
print("model f2 bic:",result2.bic)
f3 = 'Purchased ~ Age+EstimatedSalary'
logits3 = smf.logit(formula = f3, data = cst)
result3 = logits3.fit()
print(result3.summary())
print("model f3 aic:",result3.aic)
print("model f3 bic:",result3.bic)
f4 = 'Purchased ~ Gender+EstimatedSalary'
logits4 = smf.logit(formula = f4, data = cst)
result4 = logits4.fit()
print(result4.summary())
print("model f4 aic:",result4.aic)
print("model f4 bic:",result4.bic)






# ### Exercise 3

# In[31]:



print(cst)
f3 = 'Purchased ~ Age+EstimatedSalary'
logits3 = smf.logit(formula = f3, data = cst)
result3 = logits3.fit()
print(result3.summary())
print('e^0.2335=',np.exp(0.2335))
print('e^0.0000359=',np.exp(0.0000359))

### I would say model three which is the logistic model on 'Purchased ~ Age+EstimatedSalary' is the best model
### This model has the smallet AIC and BIC, also in in the full model, we can see the p-value for variable "gender" is 0.274 which is much bigger than others and we can suggest this variable is insignificant
### Variable Age and EstimatedSalary both are numerical variable
### Interpretation on variable Age's coefficient: 
### An increase of 1 unit in Age multiplies the odds of purchase  by 26.30%.
### An increase of 1 unit in EstimatedSalary multiplies the odds of purchase by 0.00359%.
### For the intercept = -12.4340 which is interpreted as the log odds of a person with age=0 and EstimatedSalary=0 being to purchase.


# ### Exercise 4
# In this case, I think accuracy is not a good measurement metric to judge the model. Since accuracy only can be calculated in out-sample model; however, our case is a in-sample one
# 
# 

# ### Exercise 5

# In[4]:


from statsmodels.graphics.factorplots import interaction_plot


# In[13]:


###Plot the interactions of the ‘Age’ and ‘Gender’ features with the ‘Purchased’ output. 
age = cst.Age
purchased = cst.Purchased
gender = cst.Gender
fig = interaction_plot(age, gender, purchased)


# ### Exercise 6
# 
# (1) I think the regression equation should include the interactions terms
# (2) From plot (a) and (b), we can clearly see that three/two lines are not parelle with one another, do not share the same slope value
# The 'In-equal' slope indicates the effect of 'Average saving' on 'Likehood on buying a house' is not the same for all values in 'Income' lines
#  And we can infer different 'Average saving' * 'Income' match a specific unit of 'Likehood on buying a house', which shows it is necessary to have interaction terms
# 
# 
# 

# In[ ]:





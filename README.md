
# Introduction

I found this dataset on the UC Irvine Machine Learning Repository. The data is split into two categories: red wine and white wine. Each set per category consists of several chemical properties of a given wine along with a quality score from 1 to 10. In this notebook I want to play around with some statistical quantities and use some machine learning algorithms to predict scores or groups of scores.

We will adopt OSEMN pipeline strategy:

1. Obtain the data
2. Scrubbing or cleaning the data. This includes data imputation (filling in missing values) and adjusting column names.
3. Explore the data. Look for outliers or weird data. Explore the relationship between features and output varaibles. Construct a correlation matrix.
4. Model the data (ML, etc).
5. iNterpret the data. What conclusions can we make? What are the most important factors (features)? How are the varaibles related to each other?  





```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

from sklearn.neural_network import MLPClassifier

%matplotlib inline
```

# Obtain The Data
This is easy. The data is in a CSV, so we can read it in with pandas.




```python
red_df = pd.read_csv("winequality-red.csv",sep=';')
white_df = pd.read_csv("winequality-white.csv",sep=';')
```

# Scrub The Data
The data as given is already quite pristine. We could rename the columns, but the names as they are now are perfectly fine. Maybe once we get into the ML part.

# Explore the Data
First, let's look at the metadata.


```python
red_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 12 columns):
    fixed acidity           1599 non-null float64
    volatile acidity        1599 non-null float64
    citric acid             1599 non-null float64
    residual sugar          1599 non-null float64
    chlorides               1599 non-null float64
    free sulfur dioxide     1599 non-null float64
    total sulfur dioxide    1599 non-null float64
    density                 1599 non-null float64
    pH                      1599 non-null float64
    sulphates               1599 non-null float64
    alcohol                 1599 non-null float64
    quality                 1599 non-null int64
    dtypes: float64(11), int64(1)
    memory usage: 150.0 KB



```python
white_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4898 entries, 0 to 4897
    Data columns (total 12 columns):
    fixed acidity           4898 non-null float64
    volatile acidity        4898 non-null float64
    citric acid             4898 non-null float64
    residual sugar          4898 non-null float64
    chlorides               4898 non-null float64
    free sulfur dioxide     4898 non-null float64
    total sulfur dioxide    4898 non-null float64
    density                 4898 non-null float64
    pH                      4898 non-null float64
    sulphates               4898 non-null float64
    alcohol                 4898 non-null float64
    quality                 4898 non-null int64
    dtypes: float64(11), int64(1)
    memory usage: 459.3 KB


No missing data points, that's good. We see that there are about three times as many white data points as there are red.


```python
red_df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.319637</td>
      <td>0.527821</td>
      <td>0.270976</td>
      <td>2.538806</td>
      <td>0.087467</td>
      <td>15.874922</td>
      <td>46.467792</td>
      <td>0.996747</td>
      <td>3.311113</td>
      <td>0.658149</td>
      <td>10.422983</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.741096</td>
      <td>0.179060</td>
      <td>0.194801</td>
      <td>1.409928</td>
      <td>0.047065</td>
      <td>10.460157</td>
      <td>32.895324</td>
      <td>0.001887</td>
      <td>0.154386</td>
      <td>0.169507</td>
      <td>1.065668</td>
      <td>0.807569</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.600000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.990070</td>
      <td>2.740000</td>
      <td>0.330000</td>
      <td>8.400000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.100000</td>
      <td>0.390000</td>
      <td>0.090000</td>
      <td>1.900000</td>
      <td>0.070000</td>
      <td>7.000000</td>
      <td>22.000000</td>
      <td>0.995600</td>
      <td>3.210000</td>
      <td>0.550000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.900000</td>
      <td>0.520000</td>
      <td>0.260000</td>
      <td>2.200000</td>
      <td>0.079000</td>
      <td>14.000000</td>
      <td>38.000000</td>
      <td>0.996750</td>
      <td>3.310000</td>
      <td>0.620000</td>
      <td>10.200000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.200000</td>
      <td>0.640000</td>
      <td>0.420000</td>
      <td>2.600000</td>
      <td>0.090000</td>
      <td>21.000000</td>
      <td>62.000000</td>
      <td>0.997835</td>
      <td>3.400000</td>
      <td>0.730000</td>
      <td>11.100000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>0.611000</td>
      <td>72.000000</td>
      <td>289.000000</td>
      <td>1.003690</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
white_df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
      <td>4898.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.854788</td>
      <td>0.278241</td>
      <td>0.334192</td>
      <td>6.391415</td>
      <td>0.045772</td>
      <td>35.308085</td>
      <td>138.360657</td>
      <td>0.994027</td>
      <td>3.188267</td>
      <td>0.489847</td>
      <td>10.514267</td>
      <td>5.877909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.843868</td>
      <td>0.100795</td>
      <td>0.121020</td>
      <td>5.072058</td>
      <td>0.021848</td>
      <td>17.007137</td>
      <td>42.498065</td>
      <td>0.002991</td>
      <td>0.151001</td>
      <td>0.114126</td>
      <td>1.230621</td>
      <td>0.885639</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.800000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.009000</td>
      <td>2.000000</td>
      <td>9.000000</td>
      <td>0.987110</td>
      <td>2.720000</td>
      <td>0.220000</td>
      <td>8.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.300000</td>
      <td>0.210000</td>
      <td>0.270000</td>
      <td>1.700000</td>
      <td>0.036000</td>
      <td>23.000000</td>
      <td>108.000000</td>
      <td>0.991723</td>
      <td>3.090000</td>
      <td>0.410000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.800000</td>
      <td>0.260000</td>
      <td>0.320000</td>
      <td>5.200000</td>
      <td>0.043000</td>
      <td>34.000000</td>
      <td>134.000000</td>
      <td>0.993740</td>
      <td>3.180000</td>
      <td>0.470000</td>
      <td>10.400000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.300000</td>
      <td>0.320000</td>
      <td>0.390000</td>
      <td>9.900000</td>
      <td>0.050000</td>
      <td>46.000000</td>
      <td>167.000000</td>
      <td>0.996100</td>
      <td>3.280000</td>
      <td>0.550000</td>
      <td>11.400000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.200000</td>
      <td>1.100000</td>
      <td>1.660000</td>
      <td>65.800000</td>
      <td>0.346000</td>
      <td>289.000000</td>
      <td>440.000000</td>
      <td>1.038980</td>
      <td>3.820000</td>
      <td>1.080000</td>
      <td>14.200000</td>
      <td>9.000000</td>
    </tr>
  </tbody>
</table>
</div>



In both cases the lowest score is a 3. The highest score for the red wines is an 8, while that for the white wines is a 9. No 1's, 2's, or 10's in either case. The means are roughly equal. The standard deviation in both cases is less than 1, so the data is very concentrated in the 5-7 score region. Let's create a histogram to check.


```python
sns.distplot(red_df['quality'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x111e62668>




![png](output_12_1.png)



```python
sns.distplot(white_df['quality'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10adfecc0>




![png](output_13_1.png)


Let's construct a correlation matrix for each color.


```python
white_corr = white_df.corr()
red_corr = red_df.corr()
```


```python
#white heat map
plt.subplots(figsize=(10,10))
sns.heatmap(white_corr, mask=np.zeros_like(white_corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x112518cc0>




![png](output_16_1.png)



```python
#red heat map
plt.subplots(figsize=(10,10))
sns.heatmap(red_corr, mask=np.zeros_like(red_corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x112233748>




![png](output_17_1.png)


(Un)surprisingly quality is most strongly correlated with alcohol content. I'm curious about the differences between these tables, as opposed to the particular values in each table. For example, residual sugar and density are strongly correlated in white wines, but not so strongly correlated in red wines. Obviously red and white wines are different, but I'd like to understand exactly what that entails chemically.

# Fun Statistics Stuff
Based on what we've seen, it seems that the mean quality score is different between red wines and white wines. Is this difference real? Let's use a bootstrap approach to formalize this more carefully. Let $\bar{X}_{red}$ and $\bar{X}_{white}$ denote the data means of the empirical distributions of the red and white wines, respectively. Let $\mu_{red}$ and $\mu_{white}$ denote the actual means of the true distribution (which we don't know), and let $\bar{X^{*}}_{red}$ and $\bar{X^{*}}_{white}$ denote empirical bootstrap sample means. We'll use lower case x's to denote the corresponding random variables. To further ease the notation a bit define $\bar{ \Delta X } = \bar{X}_{red} - \bar{X}_{white} $ and $\bar{ \Delta X^{*} } = \bar{X^{*} }_{red} - \bar{X^{*} }_{white} $.

Let's construct a confidenc interval. The bootstrap principle asserts that if we construct several values of $\delta^{*} = \bar{\Delta x^{*}} - \bar{\Delta X} $, order them, take the .025 and .975 critical values (call them $\delta^{*}_{0.025}$ and $\delta^{*}_{0.975}$ ), and construct the interval $(\bar{\Delta X} - \delta^{*}_{0.025},\bar{\Delta X} - \delta^{*}_{0.975} )$, then we have a good 95% confidence interval for $\mu_{red} - \mu_{white}$.


```python
red_means= []
white_means = []

n_red = len(red_df['quality'])
n_white = len(white_df['quality'])

mu_red = np.mean(red_df['quality'])
mu_white = np.mean(white_df['quality'])

for j in range(10000):
    new_red_sample = np.random.choice(red_df['quality'], n_red, replace=True)
    new_white_sample = np.random.choice(white_df['quality'], n_white, replace=True)
    
    new_red_mean = np.mean(new_red_sample)
    new_white_mean = np.mean(new_white_sample)
    
    
    red_means.append(new_red_mean)
    white_means.append(new_white_mean)
    
deltas = sorted(np.array(red_means) - np.array(white_means) - (np.mean(red_df['quality']) - np.mean(white_df['quality'])   ))
ci_min = np.mean(red_df['quality']) - np.mean(white_df['quality']) - np.percentile(deltas, 5/2)
ci_max = np.mean(red_df['quality']) - np.mean(white_df['quality']) - np.percentile(deltas, 100 - 5/2)
print((ci_min, ci_max))
    


```

    (-0.19548232664300505, -0.28866034393688778)


So, our 95% confidence interval is $(-0.195, -0.289)$ for $\mu_{red} - \mu_{white}$, which is strong evidence that the two means are different. Note that the mean difference from our data is about $-0.24$, comfortably in our confidence interval. Evidently white wine is reviewed more favorably than red.

# Model the Data
The assignment of an integer score from 1 to 10 is a classification problem. The simplest model is probably logistic regression. It might be better to instead, say, classifying in what range the score lays (perhaps $< 4$, $>=4$ while $<7$, and $>= 7$).


```python
white_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4898 entries, 0 to 4897
    Data columns (total 12 columns):
    fixed acidity           4898 non-null float64
    volatile acidity        4898 non-null float64
    citric acid             4898 non-null float64
    residual sugar          4898 non-null float64
    chlorides               4898 non-null float64
    free sulfur dioxide     4898 non-null float64
    total sulfur dioxide    4898 non-null float64
    density                 4898 non-null float64
    pH                      4898 non-null float64
    sulphates               4898 non-null float64
    alcohol                 4898 non-null float64
    quality                 4898 non-null int64
    dtypes: float64(11), int64(1)
    memory usage: 459.3 KB



```python
white_ml_df = white_df.rename(columns={"fixed acidity":"x1", "volatile acidity":"x2", "citric acid":"x3",
                                      "residual sugar":"x4", "chlorides":"x5", "free sulfur dioxide":"x6",
                                      "total sulfur dioxide":"x7", "density":"x8", "pH":"x9", "sulphates":"x10",
                                      "alcohol":"x11","quality":"y"})
red_ml_df = red_df.rename(columns={"fixed acidity":"x1", "volatile acidity":"x2", "citric acid":"x3",
                                      "residual sugar":"x4", "chlorides":"x5", "free sulfur dioxide":"x6",
                                      "total sulfur dioxide":"x7", "density":"x8", "pH":"x9", "sulphates":"x10",
                                      "alcohol":"x11","quality":"y"})
```


```python
y_white = white_ml_df.iloc[:,11]
X_white = white_ml_df.iloc[:,:11]

y_red = red_ml_df.iloc[:,11]
X_red = red_ml_df.iloc[:,:11]
```


```python
X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(X_red, y_red, train_size=0.7, random_state=20)
X_white_train, X_white_test, y_white_train, y_white_test = train_test_split(X_white, y_white, train_size=0.7, random_state=20)
```


```python
regr = LogisticRegression(fit_intercept = True, C = 1e9)

logmodel_red = regr.fit(X_red_train, y_red_train)
y_red_predlog = logmodel_red.predict(X_red_test)
acclog_red = accuracy_score(y_red_predlog, y_red_test)
print("The score for the red wine is: " ,acclog_red)

logmodel_white = regr.fit(X_white_train, y_white_train)
y_white_predlog = logmodel_white.predict(X_white_test)
acclog_white = accuracy_score(y_white_predlog, y_white_test)
print("The score for the white wine is: " ,acclog_white)
```

    The score for the red wine is:  0.602083333333
    The score for the white wine is:  0.526530612245


If we were just randomlyguessing the score would be a good bit lower, but this still isn't great. I should instead try SVM and maybe a neural network?


```python
svm = SVC()

svm_model_red = svm.fit(X_red_train, y_red_train)
y_red_svm_pred = svm_model_red.predict(X_red_test)
acclog_red = accuracy_score(y_red_svm_pred, y_red_test)
print("The score for the red wine is: " ,acclog_red)

svm_model_white = svm.fit(X_white_train, y_white_train)
y_white_svm_pred = svm_model_white.predict(X_white_test)
acclog_white = accuracy_score(y_white_svm_pred, y_white_test)
print("The score for the white wine is: " ,acclog_white)


```

    The score for the red wine is:  0.589583333333
    The score for the white wine is:  0.556462585034


SVM isn't doing any better. What about a shallow neural network? Maybe later add skipping.


```python
clf = MLPClassifier(solver='lbfgs',activation='tanh', alpha=1e-5, hidden_layer_sizes=(16, 1), random_state=1)

nn_model_red = clf.fit(X_red_train, y_red_train) 
y_red_nn_pred = nn_model_red.predict(X_red_test)
acclog_red = accuracy_score(y_red_nn_pred, y_red_test)
print("The score for the red wine is: " ,acclog_red)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 1), random_state=1)

nn_model_white = clf.fit(X_white_train, y_white_train) 
y_white_nn_pred = nn_model_white.predict(X_white_test)
acclog_white = accuracy_score(y_white_nn_pred, y_white_test)
print("The score for the white wine is: " ,acclog_white)

```

    The score for the red wine is:  0.583333333333
    The score for the white wine is:  0.448979591837


The neural network doesn't seem to be working too well :(  Given the fact that the output is ordinal, I think metric beyond the percentage of properly classified examples would be better.

# Illustration of Central Limit Theorem
This is a simple illustration of how sample means follow normal distributions for large sample sizes. We can use the quality scores and their empirical bootstrap distributions as the starting distribution, and show that the difference in bootstrap sample means follow a gaussian distribution. For simplicity we use the variances in the data as instead of calculating sample variances (this sample t-statistic is still normal, however).


```python
red_means= []
red_variance = np.var(red_df['quality'])

white_means = []
white_variance = np.var(white_df['quality'])

n_red = len(red_df['quality'])
n_white = len(white_df['quality'])

mu_red = np.mean(red_df['quality'])
mu_white = np.mean(white_df['quality'])

for j in range(50000):
    new_red_sample = np.random.choice(red_df['quality'], n_red, replace=True)
    new_white_sample = np.random.choice(white_df['quality'], n_white, replace=True)
    
    new_red_mean = np.mean(new_red_sample)
    new_white_mean = np.mean(new_white_sample)
    
    #new_red_variance = np.var(new_red_sample)
    #new_white_variance = np.var(new_white_sample)
    
    red_means.append(new_red_mean)
    white_means.append(new_white_mean)
    
    #red_variances.append(new_red_variance)
    #white_variances.append(new_white_variance)
```


```python
x = sorted((np.array(red_means) - np.array(white_means)   )/( red_variance/n_red + white_variance/n_white )**0.5)
kernel = gaussian_kde(x)
y = kernel(x)
```


```python
params = stats.norm.fit(x)
y2 = stats.norm.pdf(x, params[0], params[1])
```


```python
plt.plot(x,y, color='green')
plt.plot(x,y2, color='red')
plt.hist(x,bins=100, normed=True, color='blue')
```




    (array([  2.40851861e-04,   0.00000000e+00,   2.40851861e-04,
              7.22555582e-04,   9.63407443e-04,   9.63407443e-04,
              1.20425930e-03,   1.20425930e-03,   2.16766675e-03,
              3.13107419e-03,   2.64937047e-03,   5.05788908e-03,
              6.26214838e-03,   8.42981513e-03,   9.15237071e-03,
              1.15608893e-02,   1.46919635e-02,   1.80638896e-02,
              2.26400749e-02,   2.38443342e-02,   3.25150012e-02,
              4.67252610e-02,   4.79295203e-02,   4.19082238e-02,
              6.33440394e-02,   7.53866324e-02,   8.16487808e-02,
              9.27279664e-02,   1.11996115e-01,   1.28614894e-01,
              1.31745968e-01,   1.65706080e-01,   1.81120599e-01,
              2.14117304e-01,   2.23992231e-01,   2.38443342e-01,
              2.71199195e-01,   2.72403455e-01,   2.83964344e-01,
              3.19369567e-01,   3.37192605e-01,   3.57905865e-01,
              3.70671014e-01,   3.85844681e-01,   3.89457459e-01,
              4.02704311e-01,   3.92106829e-01,   4.09929867e-01,
              4.05112830e-01,   4.01259200e-01,   3.88253200e-01,
              3.74042940e-01,   3.59591828e-01,   3.57905865e-01,
              3.41768790e-01,   3.15034234e-01,   2.97692900e-01,
              2.88781381e-01,   2.73848566e-01,   2.34348861e-01,
              2.27605008e-01,   1.95812563e-01,   1.92922340e-01,
              1.68596303e-01,   1.43066005e-01,   1.29096597e-01,
              1.15368041e-01,   9.73041518e-02,   8.59841143e-02,
              7.46640768e-02,   6.16580764e-02,   5.66001873e-02,
              4.86520759e-02,   3.61277791e-02,   3.25150012e-02,
              2.60120010e-02,   1.99907044e-02,   1.85455933e-02,
              1.73413340e-02,   1.20425930e-02,   6.74385210e-03,
              8.91151885e-03,   6.74385210e-03,   3.37192605e-03,
              3.37192605e-03,   3.13107419e-03,   1.68596303e-03,
              2.40851861e-03,   1.44511116e-03,   7.22555582e-04,
              9.63407443e-04,   7.22555582e-04,   0.00000000e+00,
              2.40851861e-04,   0.00000000e+00,   2.40851861e-04,
              2.40851861e-04,   0.00000000e+00,   0.00000000e+00,
              2.40851861e-04]),
     array([-14.10430222, -14.02126363, -13.93822503, -13.85518644,
            -13.77214784, -13.68910925, -13.60607066, -13.52303206,
            -13.43999347, -13.35695487, -13.27391628, -13.19087768,
            -13.10783909, -13.02480049, -12.9417619 , -12.8587233 ,
            -12.77568471, -12.69264612, -12.60960752, -12.52656893,
            -12.44353033, -12.36049174, -12.27745314, -12.19441455,
            -12.11137595, -12.02833736, -11.94529877, -11.86226017,
            -11.77922158, -11.69618298, -11.61314439, -11.53010579,
            -11.4470672 , -11.3640286 , -11.28099001, -11.19795142,
            -11.11491282, -11.03187423, -10.94883563, -10.86579704,
            -10.78275844, -10.69971985, -10.61668125, -10.53364266,
            -10.45060406, -10.36756547, -10.28452688, -10.20148828,
            -10.11844969, -10.03541109,  -9.9523725 ,  -9.8693339 ,
             -9.78629531,  -9.70325671,  -9.62021812,  -9.53717953,
             -9.45414093,  -9.37110234,  -9.28806374,  -9.20502515,
             -9.12198655,  -9.03894796,  -8.95590936,  -8.87287077,
             -8.78983217,  -8.70679358,  -8.62375499,  -8.54071639,
             -8.4576778 ,  -8.3746392 ,  -8.29160061,  -8.20856201,
             -8.12552342,  -8.04248482,  -7.95944623,  -7.87640764,
             -7.79336904,  -7.71033045,  -7.62729185,  -7.54425326,
             -7.46121466,  -7.37817607,  -7.29513747,  -7.21209888,
             -7.12906028,  -7.04602169,  -6.9629831 ,  -6.8799445 ,
             -6.79690591,  -6.71386731,  -6.63082872,  -6.54779012,
             -6.46475153,  -6.38171293,  -6.29867434,  -6.21563575,
             -6.13259715,  -6.04955856,  -5.96651996,  -5.88348137,  -5.80044277]),
     <a list of 100 Patch objects>)




![png](output_37_1.png)


Note that we didn't subtract off the mean from the data in constructing this distribution; hence, the distribution mean is not zero (it's ten standard deviations away fom 0!).

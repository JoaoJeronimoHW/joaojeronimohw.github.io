---
layout: post
title:  "Enhancing Economic Forecasting: New Hire Wages, Machine Learning, and the Scottish Fiscal Commission's Mandate"
date:   15-06-2025 12:25:44 +0100
categories: jekyll update
---
# 1. The SFC's Core Methodology: A Structural Foundation

The Scottish Fiscal Commission (SFC), an independent institution responsible for economic and fiscal forecasting to accompany the Scottish Government’s Budget cycle, published a technical [report](https://fiscalcommission.scot/wp-content/uploads/2021/05/How-we-forecast-the-Scottish-Economy-May-2021.pdf) on May 2021 describing their operational motivation, methodology and underlying assumptions. The report establishes the process behind the forecasting of trend GDP – defined as the maximum amount of goods and services the economy can sustainably produce – and presents a decomposition mechanism that breaks this task into several, smaller forecasting and/or calibration exercises: namely, a list of endogenous variables comprising the unemployment rate gap, private-sector real hourly wage and household consumption. In this blog post, I want to focus on the stochastic process – an error correction model - governing private-sector real hourly wages and illustrate how two key changes might help improve the accuracy of the forecast, in a way that helps the SFC to better serve their mandate. First, following [Pissarides (2010)](https://cep.lse.ac.uk/pubs/download/dp0839.pdf) I propose the distinction between new hire and incumbent private sector real wages. While the distinction made in the report between public and private sector real wages speaks to an important structural characteristic of the economy, wages for new hires are considerably more procyclical than average wages. Hence, the underlying assumption that new hires represent rigid labour costs can be a limitation for the short-term component of the ECM [(Schaefer and Singleton, 2017)](https://www.econstor.eu/bitstream/10419/173042/1/cesifo1_wp6766.pdf). Second, I illustrate how some modern forecasting methods leveraging machine learning can outperform traditional econometric models – e.g. ARIMA – when working with short panels, which are prevalent in macroeconomic data. I illustrate the quantitative impact of each of these changes separately, using simulated data in Python, using `tensorflow`, `numpy`, `pandas` and `sklearn`.


---

# 2. Why New Hire Wages Matter: Solving the Unemployment Volatility Puzzle

Traditional models use *average wages*, which smooth wage cyclicality due to stickiness in ongoing employment contracts and composition effects. Integrating new hire wages into forecasting models offers a more accurate gauge of labor market dynamics than average wages, as new hire wages exhibit significantly higher procyclicality - declining by 2.2% for every 1 percentage point rise in unemployment, compared to just 0.67% for job stayers [(Figueiredo, 2021)](https://www.aeaweb.org/content/file?id=15757). This disparity arises because new hires (particularly job switchers) face fewer "lock-in" effects from past wage commitments, allowing wages to adjust rapidly to current economic conditions. Composition effects—such as cyclical upgrades in job-match quality during expansions—further amplify this flexibility. For instance, during downturns, firms hire workers whose skills are less mismatched, temporarily suppressing new hire wages without altering underlying wage structures. By capturing real-time marginal labor costs, new hire wages provide a clearer signal of economic turning points than lagging average wage metrics.

To illustrate the potential benefit of using a different measure of labour costs, I simulate a data set that captures the key cyclicality wage differences:

{% highlight ruby %}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.filters.hp_filter import hpfilter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import pywt

np.random.seed(42)
n = 100  # Small sample size
time = np.arange(n)

productivity_trend = 100 + 0.3*time + 2*np.sin(2*np.pi*time/24)

alpha = -0.08  # Speed of adjustment
beta = 0.80     # Long-run elasticity
gamma = 0.20    # Unemployment gap coefficient

unemployment_gap = 0.5*np.sin(2*np.pi*time/12) + np.random.normal(0, 0.1, n)
wages = np.zeros_like(productivity_trend)
wages_newhires = np.zeros_like(productivity_trend)
wages[0] = beta * productivity_trend[0]  # Start at long-run equilibrium
wages_newhires[0] = beta * productivity_trend[0]

# Enhanced ECM with differentiated rigidity
for t in range(1, n):
    # Regular wages: Stronger error-correction, weaker cyclical response
    wages[t] = wages[t-1] + 0.8*alpha*(wages[t-1] - beta*productivity_trend[t-1]) + 0.5*gamma*unemployment_gap[t]
    
    # New hire wages: Standard sensitivity
    wages_newhires[t] = wages_newhires[t-1] + 2*alpha*(wages_newhires[t-1] - beta*productivity_trend[t-1]) + 1.5*gamma*unemployment_gap[t]

productivity = productivity_trend + np.random.normal(0, 1, n)
wages += np.random.normal(0, 0.5, n)
wages_newhires += np.random.normal(0, 1, n)
{% endhighlight %}

I estimate a single-equation error correction model to forecast productivity using wage data. Critical parameters from simulation setup are informed by SFC parameterization. Cyclicality parameter differences between incumbents and new hires are informed by empirical literature cited above. Cointegration between productivity and wages is structurally guaranteed in the simulation due to the data-generating process being known - non-simulated data should be accompanied by the formal stationarity tests, such as Augmented Dickey-Fuller (ADF). 

{% highlight ruby %}
import statsmodels.api as sm

def ecm_forecast(wage_data, productivity, steps=10):
    # Converting inputs to Pandas Series
    wage_data = pd.Series(wage_data)
    productivity = pd.Series(productivity)
    
    df = pd.DataFrame({
        'prod': productivity,
        'wage': wage_data,
        'prod_lag': productivity.shift(1),
        'wage_lag': wage_data.shift(1)
    }).dropna()
    
    # Error Correction Term (ECT)
    df['ect'] = df['prod_lag'] - 0.8 * df['wage_lag']  # β=0.8 from simulation
    
    # ECM specification: Δprod_t = α + γ*ECT + δ*Δwage_t
    X = sm.add_constant(df[['ect', 'wage']])
    y = df['prod'] - df['prod_lag']  # Δprod
    
    model = sm.OLS(y, X).fit()
    
    # Dynamic forecasting
    forecasts = []
    current_prod = df['prod'].iloc[-1]
    current_ect = df['ect'].iloc[-1]
    current_wage = df['wage'].iloc[-1]
    
    for _ in range(steps):
        X_forecast = [1, current_ect, current_wage]
        delta_prod = model.predict(X_forecast)[0]
        current_prod += delta_prod
        forecasts.append(current_prod)
        current_ect = current_prod - 0.8 * current_wage
        current_wage += np.random.normal(0, 0.5)  # Simulate wage evolution
    
    return np.array(forecasts)

regular_forecast = ecm_forecast(wages, productivity)
newhire_forecast = ecm_forecast(wages_newhires, productivity)
{% endhighlight %}

We can then visualize the different accuracy of regular wages and new hire wages as productivity predictors:

![Alt text for accessibility](/assets/images/prod.jpg)

Because new hire wages are more procyclical than regular wages, they are able to better approximate short term cyclical productivity fluctuations than their full sample counterpart. In addition, forecasts using new hire wages provide earlier signals for monetary policy adjustments [(Heise, Pearce and Weber, 2025)](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr1128.pdf).

![Alt text for accessibility](/assets/images/error.jpg)

Therefore, new hire wages can offer a significant accuracy improvement in GDP trend forecasting by cutting through the noise of wage rigidity. Their real-time responsiveness to productivity and labor dynamics makes them a valuable metric for forward-looking economic analysis. Visualizations of forecast comparisons and cyclical gaps provide compelling evidence for their adoption in policy and business contexts.

---

# 3. Machine Learning & ARIMA: A Complementary Toolkit

### ARIMA's Limitations in Small Samples
The SFC's annual data (n≈20) strains traditional ARIMA:
- **Overfitting risk**: 4+ parameters vs theoretical limit √20≈4.5  
- **HP filter oversmoothing**: λ=1600 masks structural breaks

### LSTM Neural Networks: Capturing Non-Linear Dynamics
**Example: Productivity Forecasting**

{% highlight ruby %}
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
LSTM(50, activation='relu', input_shape=(3, 1)),
Dense(1)
])
model.compile(optimizer='adam', loss='mse')
{% endhighlight %}

You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

---
layout: post
title:  "Enhancing Economic Forecasting: New Hire Wages, Machine Learning, and the Scottish Fiscal Commission's Mandate"
date:   15-06-2025 12:25:44 +0100
categories: jekyll update
---
## 1. The SFC's Core Methodology: A Structural Foundation

The Scottish Fiscal Commission (SFC), an independent institution responsible for economic and fiscal forecasting to accompany the Scottish Government’s Budget cycle, published a technical [report](https://fiscalcommission.scot/wp-content/uploads/2021/05/How-we-forecast-the-Scottish-Economy-May-2021.pdf) on May 2021 describing their operational motivation, methodology and underlying assumptions. The report establishes the process behind the forecasting of trend GDP – defined as the maximum amount of goods and services the economy can sustainably produce – and presents a decomposition mechanism that breaks this task into several, smaller forecasting and/or calibration exercises: namely, a list of endogenous variables comprising the unemployment rate gap, private-sector real hourly wage and household consumption. In this blog post, I want to focus on the stochastic process – an error correction model - governing private-sector real hourly wages and illustrate how two key changes might help improve the accuracy of the forecast, in a way that helps the SFC to better serve their mandate. First, following [Pissarides (2010)](https://cep.lse.ac.uk/pubs/download/dp0839.pdf) I propose the distinction between new hire and incumbent private sector real wages. While the distinction made in the report between public and private sector real wages speaks to an important structural characteristic of the economy, wages for new hires are considerably more procyclical than average wages. Hence, the underlying assumption that new hires represent rigid labour costs can be a limitation for the short-term component of the ECM [(Schaefer and Singleton, 2017)](https://www.econstor.eu/bitstream/10419/173042/1/cesifo1_wp6766.pdf). Second, I illustrate how some modern forecasting methods leveraging machine learning can outperform traditional econometric models – e.g. ARIMA – when working with short panels, which are prevalent in macroeconomic data. I illustrate the quantitative impact of each of these changes separately, using simulated data in Python, using `tensorflow`, `numpy`, `pandas` and `sklearn`.



---

## 2. Why New Hire Wages Matter: Solving the Unemployment Volatility Puzzle

Traditional models use **average wages**, which smooth cyclicality due to:
- **Stickiness** in ongoing employment contracts  
- **Composition bias** (low-wage jobs destroyed in recessions) 


As shown in Pissarides' [Unemployment Volatility Puzzle](https://cep.lse.ac.uk/pubs/download/dp0839.pdf), this understates labour cost sensitivity to economic conditions.

### The New Hire Wage Advantage
**Key empirical findings** ([Devereux & Hart 2006](https://econpapers.repec.org/article/saeilrrev/v_3a60_3ay_3a2006_3ai_3a1_3ap_3a105-119.html)):
- New hire wages are **2-3× more procyclical** than incumbent wages  
- Construction sector new hires show **4.5% wage swing per 1% GDP change** vs 1.8% for incumbents


**Implementation in SFC Framework**:

wage_index = 0.7 * new_hire_wage + 0.3 * incumbent_wage

This better reflects **real-time labour costs**, particularly critical for:
- **Construction boom/bust cycles** (13% of Scottish GDP)  
- **Skill mismatch** adjustments post-COVID  

---

## 3. Machine Learning & ARIMA: A Complementary Toolkit

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

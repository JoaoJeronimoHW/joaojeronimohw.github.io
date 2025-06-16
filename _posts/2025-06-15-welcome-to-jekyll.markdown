---
layout: post
title:  "Enhancing Economic Forecasting: New Hire Wages, Machine Learning, and the Scottish Fiscal Commission's Mandate"
date:   15-06-2025 12:25:44 +0100
categories: jekyll update
---
## 1. The SFC's Core Methodology: A Structural Foundation

The Scottish Fiscal Commission (SFC) anchors its forecasts in a **production function framework**:

Y* = L* × PROD*

Where:
- **Y*** = Potential GDP  
- **L*** = Trend labour input (population × participation × hours)  
- **PROD*** = Trend productivity (HP-filtered output/hour)

This approach, detailed in the SFC's [2018 Technical Report](https://www.fiscalcommission.scot), combines demographic projections with cyclical adjustments using Hodrick-Prescott (HP) filters. While robust for identifying long-term trends, this methodology faces challenges capturing real-time labour market dynamics and non-linear productivity shocks.

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

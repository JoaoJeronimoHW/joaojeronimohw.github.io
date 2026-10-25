---
layout: post
title: "PPML and the Equivalence of Poisson Regression with Exponential Classification Models"
date: 2026-10-25 10:00:00 +0100
author: Joao Jeronimo
categories: [econometrics, machine learning, regression]
image: /assets/images/ppml_timing.png
description: "For a binary outcome, the Poisson pseudo-maximum likelihood estimator is exactly the same estimator as the exponential hazard MLE, coefficient for coefficient. This post derives the equivalence, explains why the Poisson tooling is orders of magnitude faster with high-dimensional fixed effects, and checks both claims numerically in Python."
---

# 1. The Problem: Exponential Models at Scale

Many problems in econometrics and machine learning require estimating how a binary event -- a worker leaving a firm, a customer churning, a transaction being fraudulent -- responds to a continuous variable such as a wage or a price. Because hazards and probabilities are necessarily positive, and because effects on rates are intrinsically multiplicative, the natural model for this is a hazard or classification model with a multiplicative (exponential) structure. In this post I am going to discuss why and how a counting model solves a binary-outcome problem and cuts compute time by orders of magnitude.

Let's look more closely at a generic exponential classification model. Suppose observation $$i$$ has a binary outcome $$y_i \in \{0,1\}$$ and a vector of covariates $$\boldsymbol{x_i}$$. An exponential conditional mean model specifies:

$$E[y_i \mid \boldsymbol{x_i}] = \exp(\boldsymbol{x_i}'\boldsymbol{\beta})$$

In the context of duration/hazard models -- used, for example, when predicting whether and when a borrower defaults, when modeling time-to-churn in subscription businesses or worker turnover -- the hazard rate for, say, worker $$i$$ separating from their firm in period $$t$$ is specified as:

$$h_i(t) = h_i = \exp(\boldsymbol{x_i}'\boldsymbol{\beta})$$

from where we can see that the exponential model imposes a constant hazard. From the constant hazard, the cumulative hazard and survivor functions follow:

$$\Lambda_i(t) = \int_0^t \lambda_i \, du = \lambda_i t, \qquad S_i(t) = \exp\!\big(-\Lambda_i(t)\big) = e^{-\lambda_i t},$$

and the density is the product of the hazard and the survivor function:

$$f_i(t) = \lambda_i(t)\, S_i(t) = \lambda_i e^{-\lambda_i t}.$$

These three pieces are all that is needed to build the likelihood contribution. An uncensored observation contributes its density $$f_i(t_i)$$, while a right-censored one contributes its survivor $$S_i(t_i)$$. Hence,

$$L_i = f_i(t_i)^{\delta_i} S_i(t_i)^{1-\delta_i}$$

where $$\delta_i$$ is the event (censoring) indicator for observation $$i$$, equal to 1 for uncensored failures and equal to 0 for spells still ongoing when we stopped looking. Substituting $$f_i = \lambda_i(t)\, S_i(t)$$ gives the compact hazard-model form (valid for any hazard shape, not just exponential):

$$L_i = \big[\lambda_i(t_i) S_i(t_i)\big]^{\delta_i} S_i(t_i)^{1-\delta_i} = \lambda_i(t_i)^{\delta_i}\, S_i(t_i).$$

Taking logs, and using $$\log S_i(t_i) = -\Lambda_i(t_i)$$:

$$\ell_i = \delta_i \log \lambda_i(t_i) - \Lambda_i(t_i).$$

From here we specialize to the exponential model using $$\lambda_i(t) = \lambda_i$$ constant, $$\Lambda_i(t_i) = \lambda_i t_i$$ and $$\lambda_i = \exp(x_i'\beta)$$:

$$\ell_i = \delta_i\,(x_i'\beta) - \exp(x_i'\beta)\, t_i.$$

Note that, with no censoring ($$\delta_i = 1$$ for all $$i$$) this collapses to the ordinary exponential-density log-likelihood $$\log \lambda_i - \lambda_i t_i$$. And the score for an intercept-only model, $$\partial \ell_i / \partial \lambda_i = \delta_i / \lambda_i - t_i$$, summed and set to zero yields $$\hat\lambda = \sum_i \delta_i / \sum_i t_i$$ -- the total number of events over total exposure time, which is the natural MLE of a constant hazard.

**So why does this break at scale?** Let's say $$n = 45{,}000{,}000$$ observations, with multiple sets of high-dimensional fixed effects (year, region, gender, education, nationality). These large volumes of feature-heavy data, increasingly more common across many industries, impose three distinct computational bottlenecks.

The first and perhaps the most intuitive one is a **memory bottleneck**. Fixed effects are traditionally encoded as dummy variables. With $$k$$ continuous covariates and $$G$$ total fixed-effect levels (say $$G = 10{,}000$$ year $$\times$$ region $$\times$$ education cells), the design matrix has $$n \times (k + G)$$ elements. At 8 bytes per float64, the memory load is **3.6 TB**! No laptop or standard server can hold this.

The second concerns **numerical instability in IRLS (iteratively reweighted least squares) used to fit GLMs**. For any GLM with linear predictor $$\eta_i = x_i'\beta$$, mean $$\mu_i = g^{-1}(\eta_i)$$ and variance function $$V(\mu_i)$$, the Fisher-scoring update $$\beta^{t+1} = (X'WX)^{-1}X'Wz$$ is exactly a WLS regression of the working response $$z$$ on $$X$$ with weights $$w$$.

Both the logistic weight $$w_i^{\mathrm{logit}} = p_i(1-p_i)$$ (logistic regression is the statistically canonical choice for a binary outcome) and the Poisson weight $$w_i^{\mathrm{Poisson}} = \mu_i = \exp(x_i'\beta)$$ (Poisson is the family for counts, at first sight a strange choice for binary outcomes!) shrink toward zero for rare-event observations -- on that score the two models are nearly identical, since $$p_i(1-p_i) \approx \mu_i \approx p_i$$ for small $$p_i$$. The decisive difference lies in how those weights enter the fixed-effect updates. Logistic regression has no closed-form FE update, so each group requires an inner Newton step

$$\Delta\alpha_g = \frac{\sum_{i\in g}\bigl(y_i-p_i\bigr)}{\sum_{i\in g} p_i\,(1-p_i)};$$

with a 7% separation rate and high-dimensional cells, the denominator is tiny or exactly zero when the cell contains no events (quasi-complete separation), and the inner solve becomes ill-conditioned or singular. Poisson replaces that iteration with the exact ratio

$$\hat\alpha_g = \log\!\Bigl(\textstyle\sum_{i\in g} y_i\Bigr) - \log\!\Bigl(\textstyle\sum_{i\in g}\exp(x_i'\beta)\Bigr),$$

with no division by a vanishing curvature, and the only degenerate case -- a cell with no separations -- is detected and dropped cleanly. The gain is in conditioning and convergence, *not* in statistical information, which is identical by construction.

---

# 2. The Solution: Pseudo-Maximum Likelihood (PPML)

So, if we have an exponential model and want to predict customer churn on millions of observations, what do we do? The solution is to use **Pseudo-Maximum Likelihood (PPML)**, which refers to maximising a Poisson log-likelihood even when the outcome is not actually Poisson-distributed. The justification for this comes from Gourieroux, Monfort and Trognon (1984), who proved that any quasi-MLE that correctly specifies the conditional mean produces consistent estimates of the mean parameters. For our exponential conditional mean, the Poisson QMLE is consistent.

But consistency actually undersells what happens here. For a *binary* outcome the Poisson estimator is not merely consistent for the right mean -- it is the *exact same estimator* as the exponential hazard MLE we started with. Not asymptotically close: identical, coefficient for coefficient, on any finite sample. To see it, write the Poisson log-likelihood contribution with mean $$\mu_i = \exp(x_i'\beta)$$,

$$\ell_i^{\text{Pois}} = y_i \log \mu_i - \mu_i - \log(y_i!),$$

and watch two terms collapse. Because $$y_i \in \{0,1\}$$, the factorial vanishes for *every* observation ($$\log 0! = \log 1! = 0$$); and $$\log \mu_i = x_i'\beta$$ by construction. What survives is

$$\ell_i^{\text{Pois}} = y_i\,x_i'\beta - \exp(x_i'\beta),$$

which is term-for-term the exponential-hazard contribution derived above (with the event indicator playing the role of $$y_i$$; a non-trivial exposure enters Poisson as an offset $$\log t_i$$ and leaves the score in $$\beta$$ untouched). So the two models are not cousins -- they are the same likelihood wearing different labels.

**The shared score equation.** Summing and differentiating gives the estimating equation for *both* models at once,

$$\frac{\partial L}{\partial \beta} = \sum_i x_i\bigl(y_i - \exp(x_i'\beta)\bigr) = 0.$$

It is worth reading this slowly, because it is the whole justification for the trick: choose $$\beta$$ so that, in every covariate direction, the fitted events $$\exp(x_i'\beta)$$ match the observed events $$y_i$$ on average. Nowhere does it ask whether $$y$$ is *really* Poisson -- it only asks that the conditional mean be right. That is precisely the pseudo-likelihood idea, and it is why we may borrow the Poisson machinery without believing the Poisson story for a moment.

**So where does the speed-up come from?** If the two estimators are numerically identical, the gain cannot live in the mathematics. Rather, it lives in the *software the maths unlocks* -- specifically, in one property that the log link has and the logistic link does not: fixed effects can be concentrated out in closed form. Suppose we absorb a group effect $$\alpha_g$$, so $$\mu_i = \exp(\alpha_g + x_i'\beta)$$. Its first-order condition is $$\sum_{i\in g}\bigl(y_i - \exp(\alpha_g + x_i'\beta)\bigr) = 0$$. Because the link is logarithmic, $$\alpha_g$$ enters as a pure multiplicative factor $$e^{\alpha_g}$$ that pulls straight out of the sum:

$$e^{\alpha_g}\sum_{i\in g}\exp(x_i'\beta) = \sum_{i\in g} y_i \quad\Longrightarrow\quad \hat\alpha_g = \log\!\sum_{i\in g} y_i - \log\!\sum_{i\in g}\exp(x_i'\beta).$$

No iteration, no inner loop: each group's effect is simply the log-ratio of its observed events to its baseline-predicted events. The intuition is very simple -- $$\hat\alpha_g$$ just rescales each cell so that its predicted and realized event counts agree, a one-shot moment match done cell by cell. The logistic link buys you none of this, because $$\sigma(\alpha_g + x_i'\beta)$$ does not factor into a piece in $$\alpha_g$$ times a piece in $$x_i'\beta$$; $$\alpha_g$$ stays trapped inside the sigmoid, and the only way out is a Newton iteration per group per pass.

**Why this slays all three bottlenecks.** The closed-form update needs nothing but two scalars per group -- $$\sum_{i\in g} y_i$$ and $$\sum_{i\in g} \exp(x_i'\beta)$$ -- so the estimator never materialises the dummy block at all. Alternating projections sweep through the fixed effects storing those group sums, bringing memory down from $$O(n \times G)$$ to $$O(n + G)$$: the 3.6 TB design matrix simply evaporates. The IRLS weights $$\exp(x_i'\beta)$$ stay strictly positive, so the working system stays well-conditioned. And each FE block is updated in a single pass rather than a Newton loop, so convergence is fast. The exponential hazard model shares every one of these properties -- it is, after all, the same likelihood -- but a generic survival or MLE routine fits it without exploiting any of them, dutifully trying to invert a system that includes the enormous fixed-effect block. The mature high-dimensional-FE engines (`fixest`, `pyfixest`, `ppmlhdfe`), grown out of the gravity-equation PPML literature, were built around exactly this concentration trick. "Poisson is faster than the exponential model" is really shorthand for "the Poisson tooling uses the structure that both models have, and the naïve exponential fit does not."

The cleanest way to convince yourself that this is no sleight of hand is to put both estimators on the same data and check two things at once: that the coefficients coincide to machine precision, and that the Poisson engine returns them in a fraction of the time. That is exactly what the next example does.

---

# 3. Numerical Example

We draw $$n = 500{,}000$$ observations from a *known* exponential conditional mean, $$E[y_i \mid x_i] = \exp(x_i'\beta)$$, with two covariates -- a continuous "log wage" and an interaction term -- and true coefficients $$\beta = (-2.5,\, 0.5,\, -0.3)$$. Each binary outcome is then a Bernoulli draw with probability $$\exp(x_i'\beta)$$. Two features of this design matter. First, because we chose $$\beta$$ ourselves, we can check not only whether the two estimators agree with *each other*, but whether they recover the *truth*. Second, the intercept of $$-2.5$$ puts the outcome rate at roughly 9% -- squarely in the rare-event regime that motivated the whole discussion, and close to the 7% separation rate of the labour data.

{% highlight python %}
import numpy as np, statsmodels.api as sm
from scipy.optimize import minimize

np.random.seed(42)
n = 500_000
X = np.column_stack([np.ones(n),
                     np.random.normal(0, 1, n),      # log wage
                     np.random.normal(0, 0.3, n)])   # interaction
true_beta = np.array([-2.5, 0.5, -0.3])
y = np.random.binomial(1, np.clip(np.exp(X @ true_beta), 0, 1), n)

# Method 1: direct MLE of the exponential-hazard log-likelihood
nll   = lambda b: -np.sum(y * (X @ b) - np.exp(X @ b))
score = lambda b: -X.T @ (y - np.exp(X @ b))
b_hazard = minimize(nll, np.zeros(3), jac=score, method="BFGS").x

# Method 2: Poisson GLM with log link (PPML)
b_ppml = sm.GLM(y, X, family=sm.families.Poisson()).fit(disp=False).params

print(np.abs(b_hazard - b_ppml).max())   # ~1e-8: the two are one estimator
{% endhighlight %}

*Direct exponential-hazard MLE versus Poisson PPML on the same data.*

We fit the same data two ways: a direct maximiser of the exponential-hazard log-likelihood (hand-rolled BFGS with the analytic score), and a Poisson GLM with a log link. The two columns are indistinguishable (see Table 1). Every coefficient agrees to eight decimal places, the largest discrepancy across all three parameters is of order $$10^{-8}$$, and the two maximised log-likelihoods match to the same precision. The only reason the difference is not *literally* zero is that the two routines stop iterating at marginally different points; it is floating-point bookkeeping. (Both estimates are also within sampling error of the true $$\beta$$, confirming consistency -- but the headline is the agreement *between* the two columns, not either one's distance from the truth.)

| Parameter | True | Exp. hazard MLE | Poisson PPML | Abs. diff |
|:----------|-----:|----------------:|-------------:|----------:|
| $$\beta_0$$ (intercept) | $$-2.5$$ | $$-2.50335$$ | $$-2.50335$$ | $$1.7\times10^{-11}$$ |
| $$\beta_1$$ (log wage) | $$0.5$$ | $$0.50577$$ | $$0.50577$$ | $$2.0\times10^{-11}$$ |
| $$\beta_2$$ (interaction) | $$-0.3$$ | $$-0.30460$$ | $$-0.30460$$ | $$5.2\times10^{-10}$$ |
| Log-likelihood | | $$-151176.26$$ | $$-151176.26$$ | $$2.9\times10^{-11}$$ |

*Table 1: Exponential hazard MLE versus Poisson PPML on n = 500,000 binary observations drawn from an exponential conditional mean (outcome rate ≈ 9.3%). The two estimators return the same coefficients to eight decimal places and maximise the same log-likelihood -- they are one estimator, not two.*

Notice, however, that the equivalence above was computed *without* fixed effects, and there the two fits take essentially the same time -- as they must, since they are solving the same small problem. The gap appears only when we add the ingredient that real data always carries: high-dimensional fixed effects. So we now grow the sample from 100,000 rows into the millions and let the number of fixed-effect cells grow with it ($$G = n/2000$$), mimicking the way more data brings more year $$\times$$ region $$\times\cdots$$ cells. We then fit the *same* estimator two ways -- encoding the fixed effects as explicit dummy columns, and absorbing them with the closed-form Poisson update from the previous section. The contrast between the two routines is the whole story:

{% highlight python %}
# (A) naive: fixed effects as explicit dummy columns, dense IRLS -- O(n x G)
def fit_dense(x1, g, y, G, tol=1e-9):
    X = np.column_stack([x1, (g[:, None] == np.arange(G)).astype(float)])
    b = np.r_[0.0, np.full(G, np.log(y.mean()))]
    while True:
        mu  = np.exp(np.clip(X @ b, -30, 30))
        XtW = X.T * mu
        new = np.linalg.solve(XtW @ X, XtW @ (X @ b + (y - mu) / mu))
        if np.abs(new - b).max() < tol: return new[0]
        b = new

# (B) PPML: fixed effect concentrated out in closed form each step -- O(n + G)
def fit_ap(x1, g, y, G, tol=1e-9):
    Xs, b = x1[:, None], np.zeros(1)
    while True:
        base = np.exp(np.clip(Xs @ b, -30, 30))
        ye   = np.bincount(g, y,    G)              # group event sums
        me   = np.bincount(g, base, G)              # group baseline sums
        la   = np.log(np.where(ye > 0, ye, np.nan)) - np.log(me)
        mu   = base * np.exp(np.nan_to_num(la, nan=-np.inf)[g])  # 0-event cells drop
        step = np.linalg.solve((Xs.T * mu) @ Xs, Xs.T @ (y - mu))
        b   += step
        if np.abs(step).max() < tol: return b[0]
{% endhighlight %}

*The two engines for one estimator: dense dummy columns versus closed-form FE absorption. (Driver loop over n and plotting omitted.)*

![Cost of high-dimensional fixed effects as n and the number of FE cells grow together](/assets/images/ppml_timing.png)

*Cost of high-dimensional fixed effects as n and the number of FE cells grow together.*

The dummy-variable fit (red) climbs steeply and then simply stops: beyond a few hundred thousand rows the design matrix no longer fits in memory -- the 3.6 TB wall of the opening section, reproduced in miniature. The absorbed PPML (blue) is a near-straight line on the log-log axes, sailing past 6.4 million rows in roughly two seconds, because it never materialises the dummy block and its footprint is only $$O(n + G)$$. And at every sample size where both can run, they return the *same* slope -- the speed costs nothing in accuracy. The two are not competing estimators with a speed--accuracy trade-off between them; they are the identical estimator, and one route to it is simply blocked by the hardware long before the other breaks a sweat.

Together, the table and the graph are the empirical face of the two claims from the previous section. The table confirms the first -- that for binary outcomes the Poisson and exponential-hazard estimators are numerically identical, not merely asymptotically equivalent. The graph confirms the second -- that the dramatic speed-up lives not in the mathematics, which the two models share, but in the software the log link unlocks: the closed-form fixed-effect concentration that lets the Poisson engine skip the dummy block entirely. "Poisson is faster," stated precisely, is "the Poisson tooling exploits structure that both models have, and the naïve exponential fit throws away."

So for binary outcomes, Poisson Pseudo-Maximum Likelihood is very much a free lunch. It returns *exactly* the coefficients of the exponential-hazard MLE -- the model we actually wanted -- while unlocking an estimation machinery that makes tens of millions of observations with thousands of fixed-effect levels tractable on ordinary hardware!

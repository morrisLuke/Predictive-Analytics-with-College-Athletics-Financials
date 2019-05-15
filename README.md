# Predictive-Analytics-with-College-Athletics-Financials
How well can different predictive analytics project athletic program spending and classify programs into Power Five, Group of Five or other?

## Executive Summary
While funding and spending for college athletic programs is a subject reported on quite often, nobody has tried to tease out predictive trends based on the figures divulged by public university programs that are subject to Freedom of Information Act requests. In this analysis, I looked at three questions to see what could be told from five years of such documents.

The first question asked how well an athletic program’s spending for a specific year could be predicted by looking at its revenue figures.  Using a linear regression analysis, I was able to pin down a program’s spending for a particular year within an average error range of $4.9 million. While that’s not a figure worth setting the next year’s budget forecast on, it’s a decent narrowing of a figure that can balloon as high as $163 million.

In the effort, some interesting trends in the revenue-to-spending ratio were also uncovered. TV revenue was the top per-dollar-spent revenue generator at $1.70. NCAA disbursements and ticket revenue followed at $1.60 and $1.40, respectively.

The second question in my analysis attempted to determine whether a school was a member of a “Power Five” conference based solely on financial figures. Logistic regression and k-nearest neighbor models were able to predict Power Five membership with 97.3 percent accuracy.

Another level of difficulty was added to the question, asking whether the model could correctly label as school as part of the “Power Five,” “Group of Five,” or “other.” The best accuracy in predicting membership came from a k-nearest neighbor model with k=1 that scored a 88.4 percent accuracy rating.

The final question asked how well can a decision tree could predict what percentage of an athletic program’s funding comes in the form of subsidies. The most accurate predictions came from a boosted tree model that was able to narrow down the subsidy percentages within 4.28 percentage points on average.

Answering the final question also uncovered which variables play the biggest part in predicting those percentages. By and far, ticket revenue played the biggest role. From there, NCAA distributions and total revenue before subsidies had a much smaller but still prominent role in predicting the percentages.
 
## About The Data

The data being examined was taken from a [Huffington Post/Chronicle of Higher Education study](https://projects.huffingtonpost.com/projects/ncaa/sports-at-any-cost) of five years’ worth of public university athletic departments’ filings on the money going in and out of the program, to and from where. In all, the dataset has 49 variables and 1,014 cases.

Some tweaks to the original dataset were made to ready it for the analysis I am doing. I eliminated entries from Utah valley State because of null values that complicated analysis. Errors discovered in Florida’s 2013 filing led to exclusion of that case. I also added in a `grouping` category to associate a team as a "Power Five," "Group of Five" or "other" program depending on its conference affiliation. An `isPowerFive` category served as a binary of whether or not a program was part of the Power Five. Also, many of the variables ended up being different variations of the same thing (e.g. ticket sales vs. inflation-adjusted ticket sales). In those cases, inflation-adjusted figures were used for analysis. Many variables that were essentially subtotals of other variables combined were also excluded from analysis. 

The largest undertaking in data adjustment came in correcting conference affiliations year by year. The original dataset listed teams by their conference for the 2014-15 academic year no matter the affiliation for any actual school year. I went back and adjusted conference (as well as grouping) to reflect each school for the specific academic year.

## Question 1: Projecting program spending

Looking at the linear model coefficients, it’s interesting to see associations of how much revenue is generated in each category for every dollar spent.  It won’t come as a surprise that TV revenue is the top per-dollar-spent revenue generator at $1.70. Another usual suspect, ticket sales, comes in at $1.40, following NCAA disbursements at $1.60. It makes sense that you’d see TV revenue as the leader considering that while ticket sales may slip in an off year, the games are still getting televised. That means the years-long TV contracts conferences sign on behalf of their teams are much less volatile when the on-field or on-court product underperforms.

A $4.9 million value for the root mean square error is not great, even in a situation where the top spending total happened to top $163 million. Some of the shapes seen in the residual plots show this isn’t the greatest set to run a linear regression on. Transforming the spending figures could fix some of the issues, but that would come at the cost of interpretability of the results. Because of that, we’re working with the figures as they are. The adjusted R-squared figure tells us that the model claims it can account for 97 percent of the variability in the results.

Cross validation shows that our model was one example that was actually on the low end of RMSE. In that process, the variability in spending estimates could be closer to $5.3 million. That’s quite a bit more than a rounding error, so don’t go planning your athletic budget on this model.

## Question 2: Determining program membership from financials

Two versions of the question were asked here: Using financial figures, can a model correctly identify which schools are Power Five programs and which aren’t? That one was answered using logistic regression, linear discriminant analysis, quadratic discriminant analysis and k-nearest neighbors regression analysis using k values of 1, 3 and 5.

Because logistic regression isn’t well-suited for questions with more than two possible results, it was left out of the analysis of the second version of the question: Can a model correctly identify which schools are Power Five programs, which are Group of Five programs, and which are “others”?

On the first version of the question, logistic regression and all three versions of the K-nearest neighbors models tied for best performance with 97.3 percent accuracy in guessing whether a school is part of the Power Five. That is a strong improvement over the baseline rate of 76.4 percent.  

**Performance on original classification question with two options**

| Model Type | Model	Accuracy | CV accuracy |
| ---------- | --------------- | ----------- |
| Logit | 97.3 | 98.3 |
| LDA | 96.7 | 97.3 |
| QDA | 96.7 | 96.4 |
| KNN1 | 97.3 | 98.3 |
| KNN3 | 97.3 | 98.3 |
| KNN5 | 97.3 | 98.3 |

On the second version of the question, the K-nearest neighbors model with k=1 stood above the rest with an accuracy of 88.4 percent. Cross validation on the knn model showed there was even more room for improvement, possibly up to 94 percent accuracy with the right k value. Baseline rate is 54.5 percent.

**Performance on “bonus” classification question with three options**

| Model Type | Model	Accuracy | CV accuracy |
| ---------- | --------------- | ----------- |
| LDA | 83.4 | 85.1 |
| QDA | 84.1 | 85.4 |
| KNN1 | 88.4 | 94.0 |
| KNN3 | 86.7 | 94.0 |
| KNN5 | 86.1 | 94.0 |

## Question 3: Pinning down subsidy proportions

The final question used different executions of decision trees to find out which variables about a program are the most vital in determining what percentage of its revenue comes from subsidies. Root mean squared error was used to determine the best-performing model, with the one boasting the lowest RMSE taking the prize.

Every model showed that ticket revenue played the biggest role in identifying a program’s subsidy rate. An initial tree was set up using all cases and nodes were not limited. In that run, the 726 programs that reported less than $6.75 million in ticket revenue were projected to have subsidy rates upward of 70 percent.  Further splits among those 726 programs were also made along ticket-revenue lines. The highest projected subsidy rate (78.6 percent) was projected to go to the 294 programs reporting ticket revenue under $439,000. Among the programs topping $6.75 million in ticket revenue, those that reported a loss worse than $9.96 million before subsidies were expected to have subsidy rates around 27 percent. For the programs less in the red (or even in the black), projected subsidy rate was just 5 percent.

Following a round of cross-validation, the basic tree was pruned to the suggested two nodes, with the split coming along at $5.3 million in ticket revenue. Projected subsidy rates for programs below that threshold were 70.6 percent while programs at or above the threshold were expected to see just 13.3 percent subsidization. When test data was run through the model, it returned a root mean squared error of 0.092, suggesting that projections were within 9.2 percentage points of the true rates.

Next, a bagged model was attempted in hopes of reducing the error rate. Optimizing a model from an aggregate of 500 trees constructed, the bagged model was able to reduce the RMSE to just 4.94 percentage points from the true subsidy rates. 
 
A random forest model was able to reduce the RMSE even further, to 4.87 percentage points. An importance plot created from the model also shows that ticket revenue is the most important variable in determining a program’s subsidy rate. While the very first tree we created suggested that net revenue before subsidy should be the second most important variable, it fell to fourth in the random forest model, behind NCAA distributions and royalties as well as ticket revenue.

The final type of decision-tree model attempted was a boosted model. It was able to get the RMSE as low as 4.28 percentage points while also confirming ticket revenue’s top spot in importance when it comes to determining subsidy rates. Shrinkages of 0.001 and 0.2 were both attempted, with the former performing better.

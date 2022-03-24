Predicting Students’ Final Grades Using Machine Learning Methods with Online Course Data
================
Dustin
3/23/2022

## Introduction

I will be following a walkthrough that applies machine learning to education. My goal is to learn more about machine learning in R and how to use it in education. The walkthrough will be from an open-access textbook about [Data Science in Education](https://datascienceineducation.com/c14.html).

From the dataset I am working with, I will predict a final grade from variables within the dataset. I am not worried about how the variables relate to the outcome I am trying to predict for this task. I will be using random forest modeling as my method to predict the final grade outcome.

## Background on the Dataset

Online learning has become a popular way to learn and to educate people. When people use online platforms, they submit quizzes, submit homework, and spend time learning in a learning management system. This process is similar to in-person teaching to facilitate learning. However, in-person instruction can rely on students' behavioral cues to help gauge whether the student is engaged in the learning environment. When measuring students ' engagement in an online course, it is more difficult for instructors to use specific behavioral cues in an online learning environment. Usually, these cues are in the form of missing class repeatedly or being distracted, but this is more difficult to measure in an online learning situation.

Although online learning lacks some of the behavioral cues from in-person instruction, instructors can collect different types of data in a learning management system to gauge student engagement. This other way is by tracking student interactions with the learning management system. For example, when a student watches a posted video, the system collects data about how long the student watches the video before pausing or logging out. Therefore, educators can find ways to support students in online environments from the data being collected like they can for in-person instruction.

I will be examining a dataset that collected data on students' educational experiences attending an online science course at a virtual middle school. I want to characterize the students' motivation to achieve and their tangible engagement with the course. The dataset has self-reported motivation data and behavioral data from the learning management system that I will use to predict a final course grade.

## Research Questions

I will explore the following four questions:

1.  Is motivation more predictive of course grades than other online engagement indicators?
2.  Which types of motivation are most predictive of achievement?
3.  Which types of trace measures are most predictive of achievement?
4.  How does a random forest compare to a simple linear model (regression)?

## Data Sources

The dataset has 499 students enrolled in an online middle school science course in 2015-2016.

Specific information in the dataset includes:

1.  A self-report survey assessing three aspects of students' motivation
2.  Log-trace data, such as data output from the learning management system
3.  Discussion board data
4.  Academic achievement data

I want to fit the simplest possible model to the data. The effectiveness in predicting an outcome is the most important thing, not the fanciness of the model.

I will be using random forest modeling, an extension of decision tree modeling. To learn more about this modeling technique, follow along in the [walkthrough](https://datascienceineducation.com/c14.html).

## Loading Packages

``` r
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.1 ──

    ## ✓ ggplot2 3.3.5     ✓ purrr   0.3.4
    ## ✓ tibble  3.1.5     ✓ dplyr   1.0.7
    ## ✓ tidyr   1.1.4     ✓ stringr 1.4.0
    ## ✓ readr   2.0.2     ✓ forcats 0.5.1

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(caret)
```

    ## Loading required package: lattice

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
library(ranger)
library(e1071)
library(tidylog)
```

    ## 
    ## Attaching package: 'tidylog'

    ## The following objects are masked from 'package:dplyr':
    ## 
    ##     add_count, add_tally, anti_join, count, distinct, distinct_all,
    ##     distinct_at, distinct_if, filter, filter_all, filter_at, filter_if,
    ##     full_join, group_by, group_by_all, group_by_at, group_by_if,
    ##     inner_join, left_join, mutate, mutate_all, mutate_at, mutate_if,
    ##     relocate, rename, rename_all, rename_at, rename_if, rename_with,
    ##     right_join, sample_frac, sample_n, select, select_all, select_at,
    ##     select_if, semi_join, slice, slice_head, slice_max, slice_min,
    ##     slice_sample, slice_tail, summarise, summarise_all, summarise_at,
    ##     summarise_if, summarize, summarize_all, summarize_at, summarize_if,
    ##     tally, top_frac, top_n, transmute, transmute_all, transmute_at,
    ##     transmute_if, ungroup

    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     drop_na, fill, gather, pivot_longer, pivot_wider, replace_na,
    ##     spread, uncount

    ## The following object is masked from 'package:stats':
    ## 
    ##     filter

``` r
library(dataedu)
```

## Importing and Viewing the data

``` r
df <- dataedu::sci_mo_with_text
```

``` r
glimpse(df)
```

    ## Rows: 606
    ## Columns: 74
    ## $ student_id            <dbl> 43146, 44638, 47448, 47979, 48797, 51943, 52326,…
    ## $ course_id             <chr> "FrScA-S216-02", "OcnA-S116-01", "FrScA-S216-01"…
    ## $ total_points_possible <dbl> 3280, 3531, 2870, 4562, 2207, 4208, 4325, 2086, …
    ## $ total_points_earned   <dbl> 2220, 2672, 1897, 3090, 1910, 3596, 2255, 1719, …
    ## $ percentage_earned     <dbl> 0.6768293, 0.7567261, 0.6609756, 0.6773345, 0.86…
    ## $ subject               <chr> "FrScA", "OcnA", "FrScA", "OcnA", "PhysA", "FrSc…
    ## $ semester              <chr> "S216", "S116", "S216", "S216", "S116", "S216", …
    ## $ section               <chr> "02", "01", "01", "01", "01", "03", "01", "01", …
    ## $ Gradebook_Item        <chr> "POINTS EARNED & TOTAL COURSE POINTS", "ATTEMPTE…
    ## $ Grade_Category        <lgl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, …
    ## $ final_grade           <dbl> 93.45372, 81.70184, 88.48758, 81.85260, 84.00000…
    ## $ Points_Possible       <dbl> 5, 10, 10, 5, 438, 5, 10, 10, 443, 5, 12, 10, 5,…
    ## $ Points_Earned         <dbl> NA, 10.00, NA, 4.00, 399.00, NA, NA, 10.00, 425.…
    ## $ Gender                <chr> "M", "F", "M", "M", "F", "F", "M", "F", "F", "M"…
    ## $ q1                    <dbl> 5, 4, 5, 5, 4, NA, 5, 3, 4, NA, NA, 4, 3, 5, NA,…
    ## $ q2                    <dbl> 4, 4, 4, 5, 3, NA, 5, 3, 3, NA, NA, 5, 3, 3, NA,…
    ## $ q3                    <dbl> 4, 3, 4, 3, 3, NA, 3, 3, 3, NA, NA, 3, 3, 5, NA,…
    ## $ q4                    <dbl> 5, 4, 5, 5, 4, NA, 5, 3, 4, NA, NA, 5, 3, 5, NA,…
    ## $ q5                    <dbl> 5, 4, 5, 5, 4, NA, 5, 3, 4, NA, NA, 5, 4, 5, NA,…
    ## $ q6                    <dbl> 5, 4, 4, 5, 4, NA, 5, 4, 3, NA, NA, 5, 3, 5, NA,…
    ## $ q7                    <dbl> 5, 4, 4, 4, 4, NA, 4, 3, 3, NA, NA, 5, 3, 5, NA,…
    ## $ q8                    <dbl> 5, 5, 5, 5, 4, NA, 5, 3, 4, NA, NA, 4, 3, 5, NA,…
    ## $ q9                    <dbl> 4, 4, 3, 5, NA, NA, 5, 3, 2, NA, NA, 5, 2, 2, NA…
    ## $ q10                   <dbl> 5, 4, 5, 5, 3, NA, 5, 3, 5, NA, NA, 4, 4, 5, NA,…
    ## $ time_spent            <dbl> 1555.1667, 1382.7001, 860.4335, 1598.6166, 1481.…
    ## $ TimeSpent_hours       <dbl> 25.91944500, 23.04500167, 14.34055833, 26.643610…
    ## $ TimeSpent_std         <dbl> -0.18051496, -0.30780313, -0.69325954, -0.148446…
    ## $ int                   <dbl> 5.0, 4.2, 5.0, 5.0, 3.8, 4.6, 5.0, 3.0, 4.2, NA,…
    ## $ pc                    <dbl> 4.50, 3.50, 4.00, 3.50, 3.50, 4.00, 3.50, 3.00, …
    ## $ uv                    <dbl> 4.333333, 4.000000, 3.666667, 5.000000, 3.500000…
    ## $ enrollment_status     <chr> "Approved/Enrolled", "Approved/Enrolled", "Appro…
    ## $ enrollment_reason     <chr> "Course Unavailable at Local School", "Course Un…
    ## $ cogproc               <dbl> 15.069737, 7.106667, 15.165854, 14.508000, 16.69…
    ## $ male                  <dbl> 0.51210526, 0.00000000, 0.11121951, 0.00000000, …
    ## $ female                <dbl> 0.16657895, 0.00000000, 0.15219512, 0.00000000, …
    ## $ friend                <dbl> 0.00000000, 0.00000000, 0.01268293, 0.00000000, …
    ## $ family                <dbl> 0.006052632, 0.000000000, 0.084878049, 0.0000000…
    ## $ social                <dbl> 6.200526, 6.140000, 5.052927, 6.133000, 7.534000…
    ## $ sad                   <dbl> 0.18078947, 0.00000000, 0.09097561, 0.00000000, …
    ## $ anger                 <dbl> 0.41868421, 0.00000000, 0.14097561, 0.10800000, …
    ## $ anx                   <dbl> 0.080000000, 0.000000000, 0.275365854, 0.7880000…
    ## $ negemo                <dbl> 1.1363158, 0.0000000, 1.4187805, 1.1520000, 1.28…
    ## $ posemo                <dbl> 3.555526, 19.010000, 2.906098, 5.591000, 3.79400…
    ## $ affect                <dbl> 4.756053, 19.010000, 4.330732, 6.743000, 5.07500…
    ## $ quant                 <dbl> 2.046842, 2.743333, 3.245366, 3.214000, 2.551000…
    ## $ number                <dbl> 0.9131579, 3.4733333, 2.3065854, 0.2570000, 0.21…
    ## $ interrog              <dbl> 1.2857895, 0.4433333, 1.7868293, 1.1030000, 1.71…
    ## $ compare               <dbl> 2.4213158, 4.1466667, 3.9021951, 2.6990000, 3.94…
    ## $ adj                   <dbl> 5.106842, 5.480000, 5.614390, 5.213000, 4.618000…
    ## $ verb                  <dbl> 18.11368, 11.02333, 16.34366, 16.31100, 17.11700…
    ## $ negate                <dbl> 1.2060526, 0.0000000, 1.6809756, 1.1300000, 0.74…
    ## $ conj                  <dbl> 5.565526, 6.660000, 5.370244, 6.203000, 7.244000…
    ## $ adverb                <dbl> 6.243421, 6.660000, 5.824878, 5.314000, 6.492000…
    ## $ auxverb               <dbl> 11.298421, 9.246667, 10.226341, 8.890000, 9.4940…
    ## $ prep                  <dbl> 12.301579, 11.850000, 12.132927, 13.626000, 12.8…
    ## $ article               <dbl> 7.828947, 2.223333, 6.767805, 9.119000, 9.830000…
    ## $ ipron                 <dbl> 6.936316, 2.743333, 5.145122, 4.335000, 7.841000…
    ## $ they                  <dbl> 1.01026316, 0.00000000, 0.84341463, 1.86300000, …
    ## $ shehe                 <dbl> 0.54342105, 0.00000000, 0.16951220, 0.00000000, …
    ## $ you                   <dbl> 1.7442105, 3.4733333, 1.1487805, 2.0490000, 2.62…
    ## $ we                    <dbl> 0.06578947, 0.00000000, 0.03317073, 0.30200000, …
    ## $ i                     <dbl> 3.646579, 7.993333, 4.689268, 3.449000, 3.142000…
    ## $ ppron                 <dbl> 7.010000, 11.470000, 6.882927, 7.662000, 6.77900…
    ## $ pronoun               <dbl> 13.98868, 14.20667, 12.02756, 12.21900, 14.61900…
    ## $ `function`            <dbl> 55.15447, 44.63000, 49.40293, 53.12700, 57.50900…
    ## $ Dic                   <dbl> 86.27895, 86.31000, 80.72220, 86.49700, 90.48700…
    ## $ Sixltr                <dbl> 20.89316, 22.20333, 20.80780, 21.80200, 15.30600…
    ## $ WPS                   <dbl> 17.413947, 9.833333, 17.922439, 18.824000, 15.66…
    ## $ Tone                  <dbl> 56.62395, 96.38000, 49.41610, 78.36900, 55.38400…
    ## $ Authentic             <dbl> 44.13079, 70.25333, 41.22366, 49.03800, 42.25000…
    ## $ Clout                 <dbl> 49.52079, 53.58333, 40.11024, 53.08800, 54.08500…
    ## $ Analytic              <dbl> 55.70316, 56.04000, 58.98098, 69.95700, 55.82000…
    ## $ WC                    <dbl> 88.31579, 34.66667, 69.34146, 61.20000, 47.10000…
    ## $ n                     <dbl> 38, 3, 41, 10, 10, 2, 21, 18, 31, 37, 37, 18, 12…

After using `glimpse(df)`, I notice that there are 74 variables and 606 observations. A lot of the variables come from discussion posts that the students created. Therefore, it is unnecessary to have all of these variables in our model. I will only select a few variables that I believe belong in our model to predict a final grade.

I am only interested in using data from one specific semester, so I will need to narrow down the data.

``` r
df <- 
  df %>%
  select(
    int,
    uv,
    pc,
    time_spent,
    final_grade,
    subject,
    enrollment_reason,
    semester,
    enrollment_status,
    cogproc,
    social,
    posemo,
    negemo,
    n
  )
```

    ## select: dropped 60 variables (student_id, course_id, total_points_possible, total_points_earned, percentage_earned, …)

## Analysis

Next, I will remove observations with missing data.

``` r
#checking how many rows are in the dataset
nrow(df)
```

    ## [1] 606

``` r
df <- na.omit(df)
```

``` r
nrow(df)
```

    ## [1] 464

After using `na.omit(df)`, I noticed that our number of rows decreased, which indicates that `na.omit` worked.

Some of the variables in the dataset will not be suitable for machine learning. The variables could be highly correlated with other variables, or there is no variability. For example, one of the variables in the current dataset has the same value for all of the observations. I will use a function to detect this variable and any others.

``` r
nearZeroVar(df, saveMetrics = TRUE)
```

    ##                   freqRatio percentUnique zeroVar   nzv
    ## int                1.314815     9.0517241   FALSE FALSE
    ## uv                 1.533333     6.4655172   FALSE FALSE
    ## pc                 1.488372     3.8793103   FALSE FALSE
    ## time_spent         1.000000   100.0000000   FALSE FALSE
    ## final_grade        1.333333    93.1034483   FALSE FALSE
    ## subject            1.648649     1.0775862   FALSE FALSE
    ## enrollment_reason  3.154762     1.0775862   FALSE FALSE
    ## semester           1.226601     0.6465517   FALSE FALSE
    ## enrollment_status  0.000000     0.2155172    TRUE  TRUE
    ## cogproc            1.000000    96.9827586   FALSE FALSE
    ## social             1.500000    96.1206897   FALSE FALSE
    ## posemo             1.000000    96.7672414   FALSE FALSE
    ## negemo            13.000000    90.7327586   FALSE FALSE
    ## n                  1.333333    10.1293103   FALSE FALSE

This function checks for zero variance, so I want to check the `zeroVar` column to see whether any variables failed this check. When a variable fails this check, the column will have an output of `TRUE`, which we see for `enrollment_status`. I see that it is "Approved/Enrolled" for all students when looking at this variable. Therefore, I will remove this variable because having no variability may cause problems for specific models.

``` r
df <-
  df %>%
  select(-enrollment_status)
```

    ## select: dropped one variable (enrollment_status)

Next, I will pre-process variables, which may be done by centering or scaling them. Another thing I want to pay attention to is the text data. We want the text data to be in a format to evaluate. Therefore, I will change the character strings into factors.

``` r
df <-
  df %>%
  mutate_if(is.character, as.factor)
```

    ## mutate_if: converted 'subject' from character to factor (0 new NA)

    ##            converted 'enrollment_reason' from character to factor (0 new NA)

    ##            converted 'semester' from character to factor (0 new NA)

Next, I will prepare the train and test datasets. But, first, I will "set the seed," which ensures that I will get the same results in the data partition if I rerun this same code.

``` r
#setting seed
set.seed(2022)

#creating a new object called trainIndex that will take 80 percent of the data
trainIndex <- createDataPartition(df$final_grade,
                                  p = .8,
                                  list = FALSE,
                                  times = 1)

#adding a new temporary variable to the dataset
#this variable will allow me to select rows according to the row number
df <-
  df %>%
  mutate(temp_id = 1:464)
```

    ## mutate: new variable 'temp_id' (integer) with 464 unique values and 0% NA

``` r
#filtering the dataset so I only get the rows indicated by the trainIndex vector
df_train <- 
  df %>%
  filter(temp_id %in% trainIndex)
```

    ## filter: removed 92 rows (20%), 372 rows remaining

``` r
#filtering dataset in a different way so that we get only the rows
#NOT in the trainIndex vector
df_test <-
  df %>%
  filter(!temp_id %in% trainIndex)
```

    ## filter: removed 372 rows (80%), 92 rows remaining

``` r
#deleting the temporary variable
df <-
  df %>%
  select(-temp_id)
```

    ## select: dropped one variable (temp_id)

``` r
df_train <- 
  df_train %>%
  select(-temp_id)
```

    ## select: dropped one variable (temp_id)

``` r
df_test <-
  df_test %>%
  select(-temp_id)
```

    ## select: dropped one variable (temp_id)

Next, I will estimate the models. I will use the train function, passing all variables in the data frame as predictors, except for the outcome variable.

There are three indicators of motivation that the virtual school measured, interest in the course (`int`), the perceived utility value of the course (`uv`), and perceived competence for the subject matter (`pc`). In addition, a couple of variables differentiate between the different courses in the dataset, the subject matter of the course (`subject`), the reason the student enrolled in the course (`enrollment_reason`), and the semester in which the course took place (`semester`). Lastly, the amount of time the student spent engaging with the online environment was measured and included as a predictor variable (`time_spent`).

The following variables are associated with the discussion board posts from the course:

-   `cogproc` - the average level of cognitive processing in the discussion board posts
-   `social` - the average level of social (rather than academic) content in the discussion board posts
-   `posemo` and `negemo` - the positive and negative emotions evident in the discussion board posts
-   `n` - the number of discussion board posts in total

I will set the seed again to ensure that our analysis is reproducible.

``` r
set.seed(2022)

#running the model
rf_fit <- train(final_grade ~.,
                data = df_train,
                method = "ranger")

#summary of the model
rf_fit
```

    ## Random Forest 
    ## 
    ## 372 samples
    ##  12 predictor
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 372, 372, 372, 372, 372, 372, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  splitrule   RMSE      Rsquared   MAE     
    ##    2    variance    15.68739  0.5164797  11.64446
    ##    2    extratrees  16.94669  0.4947767  12.14265
    ##   10    variance    14.77280  0.5335486  10.91866
    ##   10    extratrees  14.15257  0.5869209  10.45853
    ##   19    variance    14.94307  0.5254297  10.82462
    ##   19    extratrees  13.87496  0.5916400  10.26368
    ## 
    ## Tuning parameter 'min.node.size' was held constant at a value of 5
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were mtry = 19, splitrule = extratrees
    ##  and min.node.size = 5.

First, I see 372 samples, the number in the train data set. I did not specify any pre-processing steps in the model fitting. But, `preProcess` can be passed to `train()` to center, scale, and transform the data in many other ways. I used a resampling technique, which I used for selecting the tuning parameters. I can manually provide or estimate these parameters via bootstrap resampling or *k*-folds cross-validation strategies.

To interpret the findings, I want to look to minimize the Root Mean Square Error (RMSE) and maximize the variance explained (rsquared).

Therefore, the model with the value of the `mtry` tuning parameter equal to 19 seemed to explain the data best with the `splitrule` being "extratrees", and `min.node.size` held constant at a value of 5. I know this model fits best because the RMSE is the lowest (13.87), and the Rsquared is the highest of the options (.59).

The variance would be higher without resampling, and the model's predictive accuracy would be lower.

Next, I will use cross-validation as the resampling technique instead of bootstrapping to see whether I have different values.

``` r
set.seed(2022)

train_control <- 
  trainControl(method = "repeatedcv",
               number = 10,
               repeats = 10)

rf_fit1 <- 
  train(final_grade ~.,
                 data = df_train,
                 method = "ranger",
                 trControl = train_control)

rf_fit1
```

    ## Random Forest 
    ## 
    ## 372 samples
    ##  12 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold, repeated 10 times) 
    ## Summary of sample sizes: 336, 336, 335, 334, 335, 332, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  splitrule   RMSE      Rsquared   MAE      
    ##    2    variance    15.11416  0.5387604  11.252497
    ##    2    extratrees  16.33552  0.5389249  11.846487
    ##   10    variance    13.77690  0.5772063  10.298310
    ##   10    extratrees  13.41594  0.6211246   9.960667
    ##   19    variance    13.76228  0.5785718  10.152928
    ##   19    extratrees  13.13097  0.6223665   9.776303
    ## 
    ## Tuning parameter 'min.node.size' was held constant at a value of 5
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were mtry = 19, splitrule = extratrees
    ##  and min.node.size = 5.

When looking at the output, I am looking for which values of the various tuning parameters were selected. For example, in the `mtry` column, the value was 19, the split rule is "extratrees," and the minimum node size is 5. The model explored which value of `mtry` was best and whether extra tree or variance was a better split rule. Still, I forced all model iterations to a minimum node size of five.

Next, I will create my grid of values to test for `mtry` and `min.node.size`. I will stick with the default bootstrap resampling method to choose the best model. I'll randomly select some values for `mtry`, including the three used previously (2, 10, and 19). The values I'll try will be 2, 3, 7, 10, and 19.

``` r
#setting the seed
set.seed(2022)

#creating the grid of different values of mtry, different splitrules, and different min.node sizes to try
tune_grid <-
  expand.grid(
    mtry = c(2, 3, 7, 10, 19),
    splitrule = c("variance", "extratrees"),
    min.node.size = c(1, 5, 10, 15, 20)
  )

#fitting the new model using the tuning grid I created above
rf_fit2 <-
  train(final_grade ~.,
        data = df_train,
        method = "ranger",
        tuneGrid = tune_grid)

rf_fit2
```

    ## Random Forest 
    ## 
    ## 372 samples
    ##  12 predictor
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 372, 372, 372, 372, 372, 372, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  splitrule   min.node.size  RMSE      Rsquared   MAE     
    ##    2    variance     1             15.66512  0.5152384  11.61076
    ##    2    variance     5             15.69210  0.5153387  11.64114
    ##    2    variance    10             15.82789  0.5083757  11.76293
    ##    2    variance    15             15.98543  0.4991915  11.90732
    ##    2    variance    20             16.09170  0.4949206  11.97828
    ##    2    extratrees   1             16.86757  0.4966932  12.07518
    ##    2    extratrees   5             16.94269  0.4950183  12.13473
    ##    2    extratrees  10             17.16250  0.4856838  12.32750
    ##    2    extratrees  15             17.33109  0.4777599  12.46089
    ##    2    extratrees  20             17.48916  0.4736438  12.58393
    ##    3    variance     1             15.15543  0.5277161  11.28482
    ##    3    variance     5             15.20977  0.5251368  11.34725
    ##    3    variance    10             15.36579  0.5171340  11.49416
    ##    3    variance    15             15.48567  0.5094882  11.61603
    ##    3    variance    20             15.60463  0.5037785  11.70733
    ##    3    extratrees   1             15.77650  0.5324431  11.34234
    ##    3    extratrees   5             15.90698  0.5266667  11.45993
    ##    3    extratrees  10             16.13200  0.5202073  11.66739
    ##    3    extratrees  15             16.33973  0.5150479  11.81423
    ##    3    extratrees  20             16.55084  0.5076184  11.97011
    ##    7    variance     1             14.79640  0.5339147  10.98228
    ##    7    variance     5             14.82886  0.5319772  11.02977
    ##    7    variance    10             14.89726  0.5289383  11.11236
    ##    7    variance    15             14.94334  0.5274314  11.17354
    ##    7    variance    20             15.04377  0.5217244  11.26596
    ##    7    extratrees   1             14.44209  0.5778062  10.58472
    ##    7    extratrees   5             14.56423  0.5725012  10.69798
    ##    7    extratrees  10             14.69338  0.5692729  10.81290
    ##    7    extratrees  15             14.86691  0.5653392  10.94280
    ##    7    extratrees  20             15.04717  0.5592848  11.08348
    ##   10    variance     1             14.71438  0.5371978  10.86760
    ##   10    variance     5             14.71871  0.5371143  10.88034
    ##   10    variance    10             14.76243  0.5350528  10.94717
    ##   10    variance    15             14.79190  0.5341794  10.99062
    ##   10    variance    20             14.82324  0.5331739  11.04136
    ##   10    extratrees   1             14.11607  0.5871246  10.41911
    ##   10    extratrees   5             14.16155  0.5860302  10.46115
    ##   10    extratrees  10             14.25951  0.5846233  10.55862
    ##   10    extratrees  15             14.43418  0.5777406  10.68713
    ##   10    extratrees  20             14.53826  0.5756100  10.77089
    ##   19    variance     1             14.91844  0.5272296  10.81269
    ##   19    variance     5             14.93571  0.5261175  10.83998
    ##   19    variance    10             14.88949  0.5287237  10.81380
    ##   19    variance    15             14.83391  0.5322250  10.78972
    ##   19    variance    20             14.78741  0.5350361  10.80154
    ##   19    extratrees   1             13.85552  0.5919996  10.25749
    ##   19    extratrees   5             13.81944  0.5943767  10.25604
    ##   19    extratrees  10             13.88100  0.5929596  10.32364
    ##   19    extratrees  15             13.94521  0.5908010  10.37129
    ##   19    extratrees  20             14.01874  0.5885375  10.45055
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were mtry = 19, splitrule = extratrees
    ##  and min.node.size = 5.

The model with the same values as identified before for `mtry` (19) and `splitrule` (extratrees). Again the `min.node.size` is equal to 5 seems to fit best.

Next, I will look at the model more closely.

``` r
rf_fit2$finalModel
```

    ## Ranger result
    ## 
    ## Call:
    ##  ranger::ranger(dependent.variable.name = ".outcome", data = x,      mtry = min(param$mtry, ncol(x)), min.node.size = param$min.node.size,      splitrule = as.character(param$splitrule), write.forest = TRUE,      probability = classProbs, ...) 
    ## 
    ## Type:                             Regression 
    ## Number of trees:                  500 
    ## Sample size:                      372 
    ## Number of independent variables:  19 
    ## Mtry:                             19 
    ## Target node size:                 5 
    ## Variable importance mode:         none 
    ## Splitrule:                        extratrees 
    ## Number of random splits:          1 
    ## OOB prediction error (MSE):       173.8564 
    ## R squared (OOB):                  0.6126468

From this output, I see that `mtry` is equal to 19, the node size is 5, and the split rule is extra trees. Also, the OOB prediction errror (MSE) is 173.86 and the proportion of the variance explained is 0.61.

Next, I will use the test data set to be put through the model, and assign the predicted values to a column called `pred`. Also, I will create a `obs` column that includes the real final grades that students earned. Later, I will compare the predicted and observed values to see how well the model did.

``` r
set.seed(2022)

#creating a new object for the testing data
df_test_augmented <-
  df_test %>%
  mutate(pred = predict(rf_fit2, df_test),
         obs = final_grade)
```

    ## mutate: new variable 'pred' (double) with 92 unique values and 0% NA

    ##         new variable 'obs' (double) with 91 unique values and 0% NA

``` r
# transforming the new object into a data frame
defaultSummary(as.data.frame(df_test_augmented))
```

    ##       RMSE   Rsquared        MAE 
    ## 11.4677170  0.7454446  8.6089795

I can compare the above values to see how my model performs when given data not used to train the model. Comparing the RMSE values, I see that the RMSE for the train data was 13.82 and for the test data is 11.47, which is a slight improvement. In addition, the RSquared also improved a little from 0.59 to 0.75. Therefore, the model does marginally better with the test data compared to training data.

## Results

For random forest models, we can learn which variables contributed most strongly to the prediction in the model across all the trees in our forest.

I will rerun the `rf_fit2` model with the exact specifications, but I will add an argument to call the variable importance metric.

``` r
set.seed(2022)

rf_fit2_imp <-
  train(
    final_grade ~., 
    data = df_train,
    method = "ranger",
    tuneGrid = tune_grid,
    importance = "permutation"
  )

varImp(rf_fit2_imp)
```

    ## ranger variable importance
    ## 
    ##                                                       Overall
    ## n                                                   100.00000
    ## subjectFrScA                                         23.26248
    ## time_spent                                           11.52086
    ## subjectPhysA                                          6.17464
    ## semesterS216                                          2.87167
    ## pc                                                    2.70754
    ## negemo                                                1.53845
    ## cogproc                                               1.33123
    ## posemo                                                0.97868
    ## enrollment_reasonOther                                0.88034
    ## social                                                0.75130
    ## subjectOcnA                                           0.69080
    ## enrollment_reasonScheduling Conflict                  0.59299
    ## int                                                   0.20979
    ## enrollment_reasonLearning Preference of the Student   0.14624
    ## semesterT116                                          0.13377
    ## uv                                                    0.11062
    ## enrollment_reasonCredit Recovery                      0.05463
    ## subjectBioA                                           0.00000

Next, I am going to visualize the results from the above code.

``` r
varImp(rf_fit2_imp) %>%
  pluck(1) %>%
  rownames_to_column("var") %>%
  ggplot(aes(x = reorder(var, Overall), y = Overall)) +
  geom_col(fill = dataedu_colors("darkblue")) +
  coord_flip() +
  theme_dataedu()
```

![](PredictingStudentsGrades_files/figure-markdown_github/unnamed-chunk-24-1.png)

The first thing I notice is that the variable `n` is the most important. This variable is related to the discussion posts students write and how much they write in their discussion posts. The second most important is `subjectFrSca`. Forensic science is the most crucial course, and being enrolled in this course impacts the final grade. The third most important variable is the time spent in their online course.

There are subject differences in the final grade prediction for the psychology, biology, and forensic science course. In addition, the course the student enrolls in seems to affect the final grade depending on the course. Therefore, perhaps grades should be normalized within each course. Would this still be an essential predictor if I did this? However, I am not going to be diving into that next.

## Comparing Random Forest to Regression

Below, I will specify a linear model (regression) and check out how the linear model performs in predicting the real outcomes.

``` r
#making sure the variables stored as characters are converted to factors
df_train_lm <-
  df_train %>%
  mutate_if(is.character, as.factor)
```

    ## mutate_if: no changes

``` r
#creating linear regression model
lm_fit <-
  train(final_grade ~.,
        data = df_train_lm,
        method = "lm")

#append the predicted values to the training dataset for the linear model,
#so I can see both the predicted and the actual values
df_train_lm <-
  df_train %>%
  mutate(obs = final_grade,
         pred = predict(lm_fit, df_train_lm))
```

    ## mutate: new variable 'obs' (double) with 347 unique values and 0% NA

    ##         new variable 'pred' (double) with 372 unique values and 0% NA

``` r
df_train_randomfor <-
  df_train %>%
  mutate(pred = predict(rf_fit2, df_train),
         obs = final_grade)
```

    ## mutate: new variable 'pred' (double) with 372 unique values and 0% NA

    ##         new variable 'obs' (double) with 347 unique values and 0% NA

``` r
#linear model
defaultSummary(as.data.frame(df_train_lm))
```

    ##       RMSE   Rsquared        MAE 
    ## 14.1640082  0.5518145 10.5544595

``` r
#random forest
defaultSummary(as.data.frame(df_train_randomfor))
```

    ##      RMSE  Rsquared       MAE 
    ## 6.1399584 0.9347542 4.5011282

The random forest model performs better than the regression. A possible future direction is to use a more sophisticated model approach like deep learning.

## Conclusion

I have learned how to apply machine learning to an educational dataset from this walkthrough. I learned specifically to use the machine learning technique called random forest modeling. Also, I learned that this model performs better on the dataset than linear regression. From the modeling, I learned that when predicting the final grade, how much a student is writing in their discussion posts is a good predictor for their final grade. I can assume from this result if a student is writing a lot in their discussion posts. They are highly engaged in what they are learning, which influences their final grade. Therefore, the instructor might pay attention to students who do not write that much in their discussion posts to engage those students more in what they are learning.

# 📊 Occupation Prediction Using Machine Learning

**STAT5003 Statistical Data Mining Project**  
**Group:** W06G08  
**Dataset:** 1994 US Census Data

A comprehensive statistical analysis and machine learning project predicting individual occupation categories based on socio-economic and demographic characteristics using multiple classification algorithms.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation & Requirements](#installation--requirements)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results Summary](#results-summary)
- [Usage Instructions](#usage-instructions)
- [Key Findings](#key-findings)
- [Future Improvements](#future-improvements)

---

## 🎯 Project Overview

### Objective

Predict an individual's **occupation category** using socio-economic and demographic attributes through multi-class classification methods.

### Business Value

- ✅ **Career Recommendations** - Suggest suitable occupations based on personal characteristics
- ✅ **Educational Planning** - Formulate training programs aligned with occupation requirements
- ✅ **Workforce Analysis** - Understand relationships between demographics and career paths
- ✅ **Policy Making** - Support evidence-based employment and education policies

### Research Question

**Can we accurately classify individuals into occupation categories using their demographic and socio-economic features?**

---

## 📦 Dataset Description

### Source
**1994 US Census Bureau** data containing adult income and demographic information.

### Original Dataset Specifications

| Attribute | Value |
|-----------|-------|
| **Total Records** | 48,842 samples |
| **Features** | 15 attributes |
| **Target Variable** | `occupation` (14 original categories) |
| **Missing Values** | Present in `occupation`, `workclass`, `native.country` |
| **Duplicate Records** | Yes (removed during preprocessing) |

### Feature Types

**Numerical Variables (3):**
- `age` - Age of individual
- `education.num` - Years of education (ordinal)
- `hours.per.week` - Average working hours per week

**Categorical Variables (12):**
- `workclass` - Type of employment
- `education` - Education level (text)
- `marital.status` - Marital status
- `occupation` - **Target variable**
- `relationship` - Family relationship
- `race` - Racial background
- `gender` - Male/Female
- `native.country` - Country of origin
- `income` - Annual income bracket (>50K or ≤50K)

**Removed Variables (4):**
- `fnlwgt` - Census sampling weight (not predictive)
- `education` - Redundant with `education.num`
- `capital.gain` - 90%+ zeros
- `capital.loss` - 90%+ zeros

### Target Variable: Occupation Categories

**Original 14 Categories → Grouped into 5 Super-Categories:**

| Super-Category | Original Occupations | Interpretation |
|----------------|---------------------|----------------|
| **White-Collar** | Exec-managerial, Prof-specialty | Professional/managerial |
| **Blue-Collar** | Craft-repair, Machine-op-inspct, Transport-moving, Handlers-cleaners, Farming-fishing | Manual/technical labor |
| **Office** | Adm-clerical, Tech-support | Administrative support |
| **Sales** | Sales | Sales representatives |
| **Service** | Other-service, Protective-serv, Priv-house-serv, Armed-Forces | Service industry |

**Rationale for Grouping:**
- Reduces class imbalance
- Simplifies multi-class classification
- Maintains semantic meaning
- Improves model generalization

---

## 🛠️ Installation & Requirements

### Prerequisites

**R Version:** R 4.0.0 or higher  
**RStudio:** Recommended for running .Rmd files

### Required R Packages

```r
# Data manipulation
install.packages("dplyr")
install.packages("tidyr")
install.packages("readxl")
install.packages("reshape2")

# Visualization
install.packages("ggplot2")
install.packages("cowplot")
install.packages("gridExtra")
install.packages("corrplot")
install.packages("GGally")
install.packages("factoextra")

# Machine Learning
install.packages("caret")
install.packages("kknn")          # K-Nearest Neighbors
install.packages("MASS")          # Linear Discriminant Analysis
install.packages("e1071")         # Support functions
install.packages("randomForest")  # Random Forest
install.packages("nnet")          # Neural networks (multinomial logistic)
install.packages("glmnet")        # Regularized regression

# Reporting
install.packages("knitr")
install.packages("rmarkdown")
```

### Installation Script

```r
# Run this in R console to install all packages at once
packages <- c("ggplot2", "dplyr", "tidyr", "cowplot", "gridExtra", 
              "corrplot", "GGally", "factoextra", "readxl", "kknn", 
              "caret", "MASS", "e1071", "randomForest", "reshape2", 
              "knitr", "nnet", "glmnet")

install.packages(packages, dependencies = TRUE)
```

---

## 📁 Project Structure

```
occupation-prediction/
├── STAT5003_Project9.Rmd       # Main R Markdown analysis file
├── adult.csv                    # Dataset (1994 US Census)
├── README.md                    # This file
├── outputs/                     # Generated outputs
│   ├── STAT5003_Project9.html  # Rendered HTML report
│   ├── STAT5003_Project9.pdf   # PDF report (optional)
│   └── figures/                # Generated plots
└── data/                        # Additional data files
    └── adult_cleaned.csv       # Preprocessed dataset
```

---

## 🔬 Methodology

### 1. Data Preprocessing Pipeline

#### Step 1: Missing Value Treatment

**Strategy:**
- **Input Features** (`workclass`, `native.country`): Fill with **"Unknown"** category
  - Preserves information about missingness
  - Avoids data loss
  - May have predictive value
  
- **Target Variable** (`occupation`): **Delete** rows (2,809 samples removed)
  - Supervised learning requires labeled data
  - Imputation would introduce label noise
  - Maintains data integrity

**Result:** 48,842 → 46,033 records after missing value handling

#### Step 2: Duplicate Removal

```r
adult_data <- adult_data[!duplicated(adult_data), ]
```

**Result:** 46,033 → 45,985 unique samples

#### Step 3: Feature Engineering

**Categorical Variable Grouping:**

1. **Workclass** (7 → 4 categories)
   ```
   Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, 
   State-gov, Local-gov, Without-pay
   
   ↓
   
   Private, Self-Employed, Government, Without-pay
   ```

2. **Native Country** (41 → 3 categories)
   ```
   90%+ from United States
   
   ↓
   
   United-States, Other, Unknown
   ```

3. **Marital Status** (7 → 3 categories)
   ```
   Married-civ-spouse, Never-married, Divorced, Separated, 
   Widowed, Married-spouse-absent, Married-AF-spouse
   
   ↓
   
   Married, Single, Separated/Other
   ```

4. **Race** (5 → 2 categories)
   ```
   White (80%+), Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other
   
   ↓
   
   White, Non-White
   ```

5. **Relationship** (6 → 3 categories)
   ```
   Husband, Wife, Own-child, Other-relative, Unmarried, Not-in-family
   
   ↓
   
   In-family, Unmarried, Not-in-family
   ```

#### Step 4: Data Encoding

```r
# Convert to factor type for R modeling
categorical_vars <- c("occupation_grouped", "workclass_grouped", 
                      "native.country_grouped", "marital_grouped", 
                      "race_grouped", "relationship_grouped", 
                      "gender", "income")

adult_cleaned[categorical_vars] <- lapply(adult_cleaned[categorical_vars], as.factor)
```

---

### 2. Exploratory Data Analysis (EDA)

#### Target Variable Distribution

**Final Occupation Groups:**

| Category | Count | Percentage | Balance Status |
|----------|-------|------------|----------------|
| Blue-Collar | ~15,000 | ~33% | ✅ Largest |
| White-Collar | ~9,500 | ~21% | ✅ Well-represented |
| Office | ~8,000 | ~17% | ✅ Adequate |
| Service | ~7,500 | ~16% | ⚠️ Moderate |
| Sales | ~6,000 | ~13% | ⚠️ Smallest |

**Interpretation:**
- Reasonably balanced after grouping
- All categories have 5,000+ samples
- Sufficient for training classification models

#### Numerical Variable Insights

**Age:**
- White-Collar: Highest median age (~40 years)
- Service/Office: Younger workforce (~35 years)
- Blue-Collar/Sales: Wide age distribution

**Education Years:**
- White-Collar: Highest education (median ~13-14 years)
- Blue-Collar/Service: Lower education (median ~10 years)
- **Strong discriminative power**

**Hours per Week:**
- Sales/White-Collar: Longer hours (median ~45 hours)
- Office/Blue-Collar: Standard workweek (~40 hours)
- Extreme values (60+ hours) common in Sales

**Outlier Analysis:**
- Max age: 90 years ✅ Plausible (seniors still working)
- Max education: 16 years ✅ Valid (doctorate level)
- Max hours: 99 hours/week ✅ Possible (self-employed)
- **Decision:** No outliers removed (all values realistic)

#### Categorical Variable Relationships

**Key Findings:**

1. **Workclass × Occupation:**
   - Government → High White-Collar proportion
   - Private/Self-Employed → Dominated by Blue-Collar
   - Without-pay → Almost entirely Blue-Collar

2. **Marital Status × Occupation:**
   - Married → More Blue-Collar and White-Collar
   - Single → Higher Sales and Service proportions
   - Separated/Other → Office-oriented

3. **Gender × Occupation:**
   - Male → Strong Blue-Collar representation
   - Female → Balanced across Office, Service, Sales

#### Correlation Analysis

**Numerical Features:**

```
Pearson Correlations:
- age ↔ education.num: 0.03 (very weak)
- age ↔ hours.per.week: 0.07 (very weak)
- education.num ↔ hours.per.week: 0.15 (weak)
```

**Conclusion:** No multicollinearity concerns; all features retained.

---

## 🤖 Models Implemented

### Overview

Four supervised learning algorithms were selected for complementary strengths:

| Model | Type | Strengths | Use Case |
|-------|------|-----------|----------|
| **Multinomial Logistic Regression** | Linear | Interpretable, probabilistic | Baseline, feature importance |
| **Linear Discriminant Analysis (LDA)** | Linear | Assumes Gaussian distributions | Class separation analysis |
| **K-Nearest Neighbors (KNN)** | Non-parametric | Captures local patterns | Non-linear decision boundaries |
| **Random Forest** | Ensemble | Handles interactions, robust | Complex relationships |

---

### 1. Multinomial Logistic Regression

**Algorithm:** Softmax regression with L2 regularization (Ridge)

**Preprocessing:**
```r
# One-hot encoding of categorical variables
df_encoded <- model.matrix(~ . - occupation_grouped, data = adult_cleaned)

# Feature standardization (zero mean, unit variance)
preProc <- preProcess(train_x, method = c("center", "scale"))
```

**Hyperparameter Tuning:**

**Parameter:** `C` (inverse of regularization strength)

**Search Space:** `C ∈ {0.01, 0.1, 1, 10, 100}`

**Method:** 10-Fold Cross-Validation

**Optimization Criterion:** Minimize deviance

**Best Configuration:**
```
λ (lambda) = 0.01  →  C = 100
Interpretation: Weak regularization (model can fit data more freely)
```

**Model Training:**
```r
final_model <- glmnet(
  x, y,
  family = "multinomial",
  type.multinomial = "ungrouped",
  alpha = 0,           # Ridge (L2) regularization
  lambda = best_lambda
)
```

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 54.83% |
| **Macro Precision** | 45.66% |
| **Macro Recall** | 42.72% |
| **Macro F1-Score** | 39.84% |

**Per-Class Performance:**

| Category | Precision | Recall | F1-Score | Analysis |
|----------|-----------|--------|----------|----------|
| Blue-Collar | 0.6843 | 0.8043 | 0.6843 | ✅ Best performance |
| White-Collar | 0.6489 | 0.6892 | 0.6489 | ✅ Strong |
| Office | 0.4712 | 0.3845 | 0.4233 | ⚠️ Moderate |
| Service | 0.2377 | 0.2377 | 0.2377 | ⚠️ Weak |
| Sales | 0.0107 | 0.0203 | 0.0107 | ❌ Poor |

**Strengths:**
- Good performance on majority classes
- Interpretable coefficients
- Probabilistic predictions

**Weaknesses:**
- Struggles with minority classes (Sales)
- Linear decision boundaries
- Sensitive to class imbalance

---

### 2. Linear Discriminant Analysis (LDA)

**Algorithm:** Gaussian discriminant analysis with equal covariance assumption

**Assumptions:**
1. Each class follows a multivariate Gaussian distribution
2. All classes share the same covariance matrix
3. Features are continuous

**Training Strategy:**
```r
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

lda_model <- train(
  occupation_grouped ~ ., 
  data = train_data, 
  method = "lda", 
  trControl = ctrl
)
```

**Cross-Validation:** 10-fold repeated 3 times (30 evaluations total)

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 53.65% |
| **Macro Precision** | 42.33% |
| **Macro Recall** | 42.09% |
| **Macro F1-Score** | 39.03% |

**Per-Class Performance:**

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Blue-Collar | 0.6561 | 0.8120 | 0.6803 |
| White-Collar | 0.6222 | 0.6582 | 0.6222 |
| Office | 0.3944 | 0.3589 | 0.3758 |
| Service | 0.2238 | 0.2238 | 0.2238 |
| Sales | 0.0105 | 0.0507 | 0.0105 |

**Strengths:**
- Computationally efficient
- Works well with normal distributions
- Provides class probabilities

**Weaknesses:**
- Gaussian assumption may not hold
- Linear separability assumption
- Poor on Sales category (extreme imbalance)

---

### 3. K-Nearest Neighbors (KNN)

**Algorithm:** Distance-based non-parametric classifier

**Preprocessing:**
```r
# One-hot encoding
adult_encoded <- model.matrix(~ . - occupation_grouped - 1, data = adult_cleaned)

# Feature scaling (critical for distance-based methods)
train_data_scaled <- scale(train_data)
test_data_scaled <- scale(test_data, 
                          center = attr(train_data_scaled, "scaled:center"),
                          scale = attr(train_data_scaled, "scaled:scale"))
```

**Hyperparameter Tuning:**

**Search Grid:**
- `k` values: {61, 63, 65, ..., 81} (odd numbers only)
- Distance metrics: Manhattan (L1), Euclidean (L2)

**Method:** 10-Fold Cross-Validation

**Optimization Results:**

| Distance | Best k | CV Accuracy |
|----------|--------|-------------|
| Manhattan | 81 | **53.85%** ✅ |
| Euclidean | 81 | 53.74% |

**Best Configuration:**
```
k = 81 neighbors
distance = Manhattan (L1)
Reasoning: More robust to outliers than Euclidean
```

**Final Model Training:**
```r
final_knn_model <- kknn(
  occupation_grouped ~ .,
  train = train_df,
  test = test_df,
  k = 81,
  distance = 1  # Manhattan
)
```

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 53.82% |
| **Macro Precision** | 45.42% |
| **Macro Recall** | 43.38% |
| **Macro F1-Score** | 41.91% |

**Per-Class Performance:**

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Blue-Collar | 0.6720 | 0.7854 | 0.6720 |
| White-Collar | 0.6281 | 0.6692 | 0.6281 |
| Office | 0.4496 | 0.3679 | 0.4045 |
| Service | 0.2628 | 0.2628 | 0.2628 |
| Sales | 0.0681 | 0.0388 | 0.0681 |

**Strengths:**
- No assumptions about data distribution
- Captures local patterns
- Non-linear decision boundaries

**Weaknesses:**
- Computationally expensive at prediction time
- Curse of dimensionality
- Requires careful tuning of k

---

### 4. Random Forest

**Algorithm:** Ensemble of decision trees with class weighting

**Class Imbalance Handling:**
```r
# Inverse frequency weighting
class_counts <- table(train_data$occupation_grouped)
class_weights <- 1 / class_counts
class_weights <- class_weights / sum(class_weights)
```

**Weighted Categories:**

| Category | Sample Count | Weight |
|----------|--------------|--------|
| Blue-Collar | ~12,000 | 0.136 |
| White-Collar | ~7,600 | 0.213 |
| Office | ~6,400 | 0.253 |
| Service | ~6,000 | 0.270 |
| Sales | ~4,800 | 0.338 ✅ Highest |

**Hyperparameter Tuning:**

**Parameter:** `mtry` (number of features sampled per split)

**Search Space:** `mtry ∈ {1, 2, 3, ..., p}` where p = total features

**Method:** 10-Fold Cross-Validation with class weights

**Results:**

| mtry | CV Error |
|------|----------|
| 1 | 0.6243 |
| 3 | 0.5892 |
| **6** | **0.5637** ✅ |
| 9 | 0.5701 |
| 12 | 0.5845 |

**Best Configuration:**
```
mtry = 6 features per split
ntree = 500 trees
class weights applied
```

**Model Training:**
```r
rf_model <- randomForest(
  occupation_grouped ~ ., 
  data = train_data, 
  mtry = 6,
  ntree = 500,
  classwt = class_weights_named
)
```

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 44.17% ⚠️ |
| **Macro Precision** | 42.40% |
| **Macro Recall** | 41.79% |
| **Macro F1-Score** | 41.24% |

**Per-Class Performance:**

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Blue-Collar | 0.5589 | 0.6234 | 0.5589 |
| White-Collar | 0.5450 | 0.5583 | 0.5450 |
| Office | 0.3892 | 0.3443 | 0.3657 |
| Service | 0.3281 | 0.3281 | 0.3281 |
| Sales | 0.2056 | 0.3351 | 0.2568 |

**Strengths:**
- Handles class imbalance with weighting
- Captures non-linear interactions
- Feature importance analysis
- Robust to overfitting

**Weaknesses:**
- Lower overall accuracy than other models
- Computationally intensive
- Less interpretable (black box)

**Note:** Lower accuracy due to aggressive focus on minority classes

---

## 📈 Results Summary

### Model Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Best For |
|-------|----------|-----------------|--------------|----------|----------|
| **Logistic Regression** | **54.83%** ✅ | 45.66% | 42.72% | 39.84% | Overall accuracy, interpretability |
| **LDA** | 53.65% | 42.33% | 42.09% | 39.03% | Fast computation, baseline |
| **KNN** | 53.82% | **45.42%** | 43.38% | **41.91%** ✅ | Balanced precision/recall |
| **Random Forest** | 44.17% | 42.40% | **41.79%** | 41.24% | Minority class recall |

### Key Performance Insights

**1. Best Overall Performance:**
- **Winner:** Multinomial Logistic Regression (54.83% accuracy)
- Consistent performance across metrics
- Best for majority classes (Blue-Collar, White-Collar)

**2. Best Balanced Performance:**
- **Winner:** K-Nearest Neighbors (41.91% Macro F1)
- Highest macro F1-score indicates better class balance
- More equitable predictions across all categories

**3. Best Minority Class Handling:**
- **Winner:** Random Forest (Sales F1 = 0.2568)
- Class weighting strategy improves minority performance
- Trade-off: Lower overall accuracy

**4. Computational Efficiency:**
- **Fastest:** LDA (linear algebra operations)
- **Slowest:** KNN (distance calculations for each prediction)

---

### Per-Class Analysis

**Blue-Collar (Majority Class):**
- All models perform well (F1 > 0.55)
- Best: Logistic Regression (F1 = 0.6843)
- **Reason:** Abundant training examples

**White-Collar:**
- Consistently strong across models (F1 > 0.54)
- Best: Logistic Regression (F1 = 0.6489)
- **Reason:** Distinct feature patterns

**Office:**
- Moderate performance (F1 ~ 0.37-0.42)
- **Challenge:** Feature overlap with White-Collar

**Service:**
- Weak performance (F1 ~ 0.22-0.33)
- **Challenge:** Heterogeneous group, class imbalance

**Sales (Minority Class):**
- Poor performance across all models (F1 < 0.26)
- Best: Random Forest (F1 = 0.2568)
- **Challenge:** Smallest class, overlapping features

---

## 🚀 Usage Instructions

### Running the Analysis

**Option 1: RStudio (Recommended)**

1. Open `STAT5003_Project9.Rmd` in RStudio
2. Install required packages (see Installation section)
3. Click **"Knit"** button to generate HTML report
4. Select output format: HTML, PDF, or Word

**Option 2: R Console**

```r
# Set working directory
setwd("/path/to/project")

# Load required libraries
library(rmarkdown)

# Render the report
render("STAT5003_Project9.Rmd", output_format = "html_document")

# Or render all formats
render("STAT5003_Project9.Rmd", output_format = "all")
```

**Option 3: Command Line**

```bash
# Navigate to project directory
cd /path/to/project

# Render HTML
Rscript -e "rmarkdown::render('STAT5003_Project9.Rmd', output_format='html_document')"

# Render PDF (requires LaTeX)
Rscript -e "rmarkdown::render('STAT5003_Project9.Rmd', output_format='pdf_document')"
```

### Modifying the Analysis

**Change Train/Test Split:**
```r
# Locate this line in the code (around line 200)
split_index <- createDataPartition(df_encoded$occupation_grouped, p = 0.8, list = FALSE)

# Change to 70/30 split
split_index <- createDataPartition(df_encoded$occupation_grouped, p = 0.7, list = FALSE)
```

**Adjust Model Hyperparameters:**

**Logistic Regression:**
```r
# Change regularization values
C_values <- c(0.01, 0.1, 1, 10, 100)  # Modify these
```

**KNN:**
```r
# Change k range
k_values <- seq(61, 81, by = 2)  # Try different ranges
```

**Random Forest:**
```r
# Change number of trees
ntree = 500  # Increase for better performance (slower)
```

---

## 🔍 Key Findings

### 1. Feature Importance

**Most Discriminative Features:**
1. **Education Years** (`education.num`)
   - Strong separator between White-Collar and Blue-Collar
   - Higher education → Professional occupations
   
2. **Age**
   - Career progression indicator
   - White-Collar workers tend to be older
   
3. **Workclass**
   - Government workers → More White-Collar
   - Private/Self-employed → More Blue-Collar

4. **Gender**
   - Males → Higher Blue-Collar representation
   - Females → More balanced across Office/Service/Sales

5. **Marital Status**
   - Married → More stable careers (White/Blue-Collar)
   - Single → Higher Service/Sales proportions

**Least Discriminative:**
- Native country (90% from USA)
- Income (moderate correlation)

---

### 2. Class Imbalance Impact

**Findings:**
- Even after grouping, imbalance persists
- Sales category remains challenging (13% of data)
- All models struggle with minority classes

**Mitigation Strategies Tested:**
✅ **Grouping categories** - Reduced from 14 to 5 classes  
✅ **Class weighting** - Applied in Random Forest  
⚠️ **Regularization** - Helped generalization but not balance

**Future Solutions:**
- SMOTE (Synthetic Minority Over-sampling Technique)
- Cost-sensitive learning
- Ensemble methods with balanced bagging

---

### 3. Model Trade-offs

**Accuracy vs. Balance:**
- Logistic Regression: High accuracy, poor minority recall
- Random Forest: Lower accuracy, better minority recall
- **Recommendation:** Choose based on use case priority

**Interpretability vs. Performance:**
- Logistic/LDA: Interpretable, moderate performance
- KNN/RF: Black box, similar/lower performance
- **Recommendation:** Use Logistic for business insights

**Speed vs. Accuracy:**
- LDA: Fast, decent accuracy
- KNN: Slow, comparable accuracy
- **Recommendation:** LDA for production at scale

---

## 🎯 Recommendations

### For Practitioners

**1. Use Logistic Regression for:**
- Baseline model and feature selection
- Interpretable insights for stakeholders
- Production systems requiring fast inference

**2. Use KNN for:**
- Capturing local non-linear patterns
- Exploratory analysis
- When computational resources allow

**3. Use Random Forest with Class Weighting for:**
- Improving minority class predictions
- Feature interaction analysis
- When interpretability is not critical

**4. Avoid:**
- Single-model reliance (use ensemble voting)
- Ignoring class imbalance (apply techniques)
- Over-tuning on majority classes

### For Further Research

**Feature Engineering:**
- Create interaction terms (age × education)
- Polynomial features for non-linearity
- Domain-specific features (career stage indicator)

**Advanced Techniques:**
- **Gradient Boosting** (XGBoost, LightGBM)
- **Neural Networks** with class balancing
- **Stacking/Ensemble** methods

**Data Augmentation:**
- SMOTE for minority classes
- Combine with other census years
- External data sources (occupation requirements)

---

## 🚧 Limitations & Future Work

### Current Limitations

1. **Data Age:** 1994 Census data (30+ years old)
   - Economic landscape has changed significantly
   - New occupation types emerged (e.g., data scientists)

2. **Class Imbalance:** Persists even after grouping
   - Sales category underrepresented
   - Affects minority class predictions

3. **Feature Set:** Limited socio-economic variables
   - No geographic granularity
   - Missing skills/certifications data

4. **Temporal Dynamics:** Static snapshot
   - Cannot model career transitions
   - No longitudinal analysis

### Future Improvements

**Short-term:**
- [ ] Implement SMOTE for class balancing
- [ ] Try gradient boosting models (XGBoost)
- [ ] Create stacked ensemble model
- [ ] Add feature selection analysis
- [ ] Cross-validate on updated census data

**Long-term:**
- [ ] Collect recent data (2020+ Census)
- [ ] Incorporate job market trends
- [ ] Build career transition models
- [ ] Deploy as web application
- [ ] Add explainability (SHAP values)

---

## 📚 References

### Dataset
- **UCI Machine Learning Repository:** [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Original Source:** U.S. Census Bureau, 1994

### Methodology
- **Logistic Regression:** Hosmer, D. W., & Lemeshow, S. (2000). *Applied Logistic Regression*
- **LDA:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- **KNN:** Cover, T., & Hart, P. (1967). "Nearest neighbor pattern classification"
- **Random Forest:** Breiman, L. (2001). "Random Forests" *Machine Learning*

### R Packages
- **caret:** Kuhn, M. (2020). *Classification and Regression Training*
- **glmnet:** Friedman, J., Hastie, T., & Tibshirani, R. (2010)
- **randomForest:** Liaw, A., & Wiener, M. (2002)

---

## 👥 Contributors

**Group:** W06G08  
**Course:** STAT5003 - Statistical Data Mining  
**Institution:** University of Sydney (presumed based on course code)

---

## 📄 License

This project is an academic submission for STAT5003. 

**Usage Restrictions:**
- ❌ Do not submit as your own coursework
- ❌ Do not redistribute without permission
- ✅ Use for learning statistical data mining
- ✅ Reference for R programming and ML techniques
- ✅ Study data preprocessing best practices

---

## 📧 Contact

For questions about this project or methodology, please refer to:
- Course instructors (STAT5003)
- Original contributors (Group W06G08)

---

**📊 Built with R | Statistical Excellence | Machine Learning Best Practices**

*Last Updated: 2024*

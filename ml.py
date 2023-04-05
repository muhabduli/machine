{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "wNvnsrRKp1k8"
   },
   "source": [
    "## **Problem Statement**\n",
    "\n",
    "\n",
    "Financial inclusion remains one of the main obstacles to economic and human development in Africa. For example, across Kenya, Rwanda, Tanzania, and Uganda only 9.1 million adults (or 14% of adults) have access to or use a commercial bank account.\n",
    "\n",
    "Traditionally, access to bank accounts has been regarded as an indicator of financial inclusion. Despite the proliferation of mobile money in Africa, and the growth of innovative fintech solutions, banks still play a pivotal role in facilitating access to financial services. Access to bank accounts enable households to save and make payments while also helping businesses build up their credit-worthiness and improve their access to loans, insurance, and related services. Therefore, access to bank accounts is an essential contributor to long-term economic growth.\n",
    "\n",
    "The objective of this competition is to create a machine learning model to predict which individuals are most likely to have or use a bank account. The models and solutions developed can provide an indication of the state of financial inclusion in Kenya, Rwanda, Tanzania and Uganda, while providing insights into some of the key factors driving individuals’ financial security."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xrWCZ1rlqFZU"
   },
   "source": [
    "## **Evaluation**\n",
    "The evaluation metric for this challenge is `Mean Absolute error`, where 1 indicates that the individual does have a bank account and 0 indicates that they do not."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "DG_A60UGqSoj"
   },
   "source": [
    "## **DATA**\n",
    "\n",
    "- **Train.csv:** contains the data to be used\n",
    "\n",
    "- **VariableDefinition.csv**:Full list of variables and their explanations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {
    "id": "EIH13Fnup0ND"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sb\n",
    "from matplotlib import pyplot as plt #visualization the data\n",
    "from sklearn.model_selection import train_test_split  # split the data into trian and test sets\n",
    "# different algorithms for comparisons\n",
    "from sklearn.ensemble import RandomForestClassifier # also used for feature selection\n",
    "from sklearn.ensemble import GradientBoostingClassifier #boosting algorithm\n",
    "from sklearn.tree import DecisionTreeClassifier     \n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC   # support vector classifier (SVC)\n",
    "from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler\n",
    "from sklearn.metrics import mean_absolute_error as MAE\n",
    "import sklearn\n",
    "from sklearn import metrics\n",
    "from sklearn.metrics import get_scorer_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['accuracy',\n",
       " 'adjusted_mutual_info_score',\n",
       " 'adjusted_rand_score',\n",
       " 'average_precision',\n",
       " 'balanced_accuracy',\n",
       " 'completeness_score',\n",
       " 'explained_variance',\n",
       " 'f1',\n",
       " 'f1_macro',\n",
       " 'f1_micro',\n",
       " 'f1_samples',\n",
       " 'f1_weighted',\n",
       " 'fowlkes_mallows_score',\n",
       " 'homogeneity_score',\n",
       " 'jaccard',\n",
       " 'jaccard_macro',\n",
       " 'jaccard_micro',\n",
       " 'jaccard_samples',\n",
       " 'jaccard_weighted',\n",
       " 'matthews_corrcoef',\n",
       " 'max_error',\n",
       " 'mutual_info_score',\n",
       " 'neg_brier_score',\n",
       " 'neg_log_loss',\n",
       " 'neg_mean_absolute_error',\n",
       " 'neg_mean_absolute_percentage_error',\n",
       " 'neg_mean_gamma_deviance',\n",
       " 'neg_mean_poisson_deviance',\n",
       " 'neg_mean_squared_error',\n",
       " 'neg_mean_squared_log_error',\n",
       " 'neg_median_absolute_error',\n",
       " 'neg_negative_likelihood_ratio',\n",
       " 'neg_root_mean_squared_error',\n",
       " 'normalized_mutual_info_score',\n",
       " 'positive_likelihood_ratio',\n",
       " 'precision',\n",
       " 'precision_macro',\n",
       " 'precision_micro',\n",
       " 'precision_samples',\n",
       " 'precision_weighted',\n",
       " 'r2',\n",
       " 'rand_score',\n",
       " 'recall',\n",
       " 'recall_macro',\n",
       " 'recall_micro',\n",
       " 'recall_samples',\n",
       " 'recall_weighted',\n",
       " 'roc_auc',\n",
       " 'roc_auc_ovo',\n",
       " 'roc_auc_ovo_weighted',\n",
       " 'roc_auc_ovr',\n",
       " 'roc_auc_ovr_weighted',\n",
       " 'top_k_accuracy',\n",
       " 'v_measure_score']"
      ]
     },
     "execution_count": 155,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "get_scorer_names()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv('C:/Users/Al Amin/Downloads/NITDA_AI_Class-main/NITDA_AI_Class-main/ML_Compete/Classification/Train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>country</th>\n",
       "      <th>year</th>\n",
       "      <th>uniqueid</th>\n",
       "      <th>bank_account</th>\n",
       "      <th>location_type</th>\n",
       "      <th>cellphone_access</th>\n",
       "      <th>household_size</th>\n",
       "      <th>age_of_respondent</th>\n",
       "      <th>gender_of_respondent</th>\n",
       "      <th>relationship_with_head</th>\n",
       "      <th>marital_status</th>\n",
       "      <th>education_level</th>\n",
       "      <th>job_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Kenya</td>\n",
       "      <td>2018</td>\n",
       "      <td>uniqueid_1</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Rural</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3</td>\n",
       "      <td>24</td>\n",
       "      <td>Female</td>\n",
       "      <td>Spouse</td>\n",
       "      <td>Married/Living together</td>\n",
       "      <td>Secondary education</td>\n",
       "      <td>Self employed</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Kenya</td>\n",
       "      <td>2018</td>\n",
       "      <td>uniqueid_2</td>\n",
       "      <td>No</td>\n",
       "      <td>Rural</td>\n",
       "      <td>No</td>\n",
       "      <td>5</td>\n",
       "      <td>70</td>\n",
       "      <td>Female</td>\n",
       "      <td>Head of Household</td>\n",
       "      <td>Widowed</td>\n",
       "      <td>No formal education</td>\n",
       "      <td>Government Dependent</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Kenya</td>\n",
       "      <td>2018</td>\n",
       "      <td>uniqueid_3</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Yes</td>\n",
       "      <td>5</td>\n",
       "      <td>26</td>\n",
       "      <td>Male</td>\n",
       "      <td>Other relative</td>\n",
       "      <td>Single/Never Married</td>\n",
       "      <td>Vocational/Specialised training</td>\n",
       "      <td>Self employed</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Kenya</td>\n",
       "      <td>2018</td>\n",
       "      <td>uniqueid_4</td>\n",
       "      <td>No</td>\n",
       "      <td>Rural</td>\n",
       "      <td>Yes</td>\n",
       "      <td>5</td>\n",
       "      <td>34</td>\n",
       "      <td>Female</td>\n",
       "      <td>Head of Household</td>\n",
       "      <td>Married/Living together</td>\n",
       "      <td>Primary education</td>\n",
       "      <td>Formally employed Private</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kenya</td>\n",
       "      <td>2018</td>\n",
       "      <td>uniqueid_5</td>\n",
       "      <td>No</td>\n",
       "      <td>Urban</td>\n",
       "      <td>No</td>\n",
       "      <td>8</td>\n",
       "      <td>26</td>\n",
       "      <td>Male</td>\n",
       "      <td>Child</td>\n",
       "      <td>Single/Never Married</td>\n",
       "      <td>Primary education</td>\n",
       "      <td>Informally employed</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  country  year    uniqueid bank_account location_type cellphone_access  \\\n",
       "0   Kenya  2018  uniqueid_1          Yes         Rural              Yes   \n",
       "1   Kenya  2018  uniqueid_2           No         Rural               No   \n",
       "2   Kenya  2018  uniqueid_3          Yes         Urban              Yes   \n",
       "3   Kenya  2018  uniqueid_4           No         Rural              Yes   \n",
       "4   Kenya  2018  uniqueid_5           No         Urban               No   \n",
       "\n",
       "   household_size  age_of_respondent gender_of_respondent  \\\n",
       "0               3                 24               Female   \n",
       "1               5                 70               Female   \n",
       "2               5                 26                 Male   \n",
       "3               5                 34               Female   \n",
       "4               8                 26                 Male   \n",
       "\n",
       "  relationship_with_head           marital_status  \\\n",
       "0                 Spouse  Married/Living together   \n",
       "1      Head of Household                  Widowed   \n",
       "2         Other relative     Single/Never Married   \n",
       "3      Head of Household  Married/Living together   \n",
       "4                  Child     Single/Never Married   \n",
       "\n",
       "                   education_level                   job_type  \n",
       "0              Secondary education              Self employed  \n",
       "1              No formal education       Government Dependent  \n",
       "2  Vocational/Specialised training              Self employed  \n",
       "3                Primary education  Formally employed Private  \n",
       "4                Primary education        Informally employed  "
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 23524 entries, 0 to 23523\n",
      "Data columns (total 13 columns):\n",
      " #   Column                  Non-Null Count  Dtype \n",
      "---  ------                  --------------  ----- \n",
      " 0   country                 23524 non-null  object\n",
      " 1   year                    23524 non-null  int64 \n",
      " 2   uniqueid                23524 non-null  object\n",
      " 3   bank_account            23524 non-null  object\n",
      " 4   location_type           23524 non-null  object\n",
      " 5   cellphone_access        23524 non-null  object\n",
      " 6   household_size          23524 non-null  int64 \n",
      " 7   age_of_respondent       23524 non-null  int64 \n",
      " 8   gender_of_respondent    23524 non-null  object\n",
      " 9   relationship_with_head  23524 non-null  object\n",
      " 10  marital_status          23524 non-null  object\n",
      " 11  education_level         23524 non-null  object\n",
      " 12  job_type                23524 non-null  object\n",
      "dtypes: int64(3), object(10)\n",
      "memory usage: 2.3+ MB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "metadata": {},
   "outputs": [],
   "source": [
    "#separating the data based on categorical  and numerical features\n",
    "cat_features = ['uniqueid','location_type','cellphone_access','gender_of_respondent','relationship_with_head','marital_status','education_level','job_type']\n",
    "num_features= ['year','household_size','age_of_respondent']\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "country                   0\n",
       "year                      0\n",
       "uniqueid                  0\n",
       "bank_account              0\n",
       "location_type             0\n",
       "cellphone_access          0\n",
       "household_size            0\n",
       "age_of_respondent         0\n",
       "gender_of_respondent      0\n",
       "relationship_with_head    0\n",
       "marital_status            0\n",
       "education_level           0\n",
       "job_type                  0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 160,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# checking for outliers\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABYUAAATHCAYAAACyUhgKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAABzv0lEQVR4nOzdebiXdZ3/8df5nsMqoECyuKWSRiruoIxRRpq5VeQyM2ZjpmaJG6aSiruYggGyuIV7mWVu2TaNltNUKqKmMyruGCrIKKCibJ7v+f3RePox0oBw8Avn83hcF9fl977v7+d+c/hDrud187nrmpqamgIAAAAAQBEqtR4AAAAAAIAPjygMAAAAAFAQURgAAAAAoCCiMAAAAABAQURhAAAAAICCiMIAAAAAAAURhQEAAAAACiIKAwAAAAAURBQGAAAAAChIQ60HaK2amppSrTbVegwAAAAAoBCVSl3q6uqWe50ovJpUq02ZM+ftWo8BAAAAABSiW7d1Ul+//Chs+wgAAAAAgIKIwgAAAAAABRGFAQAAAAAKIgoDAAAAABREFAYAAAAAKEhDrQcAAAAAAFa/arWaxsZ3az0GK6m+viGVSss84ysKAwAAAEAr1tTUlDffnJMFC+bXehRWUYcOndKlS7fU1dWt0jqiMAAAAAC0Yu8F4U6duqZt23arHBT58DU1NWXx4kWZP39ukmTddbuv0nqiMAAAAAC0UtVqY3MQ7tSpS63HYRW0bdsuSTJ//tx07tx1lbaS8KI5AAAAAGilGhsbk/wtKLJ2e+/PcVX3hhaFAQAAAKCVs2VE69BSf46iMAAAAADAGqCpqelDuY8oDAAAAAAFO/DA/TNy5Dmr/T4zZ76ST35y5/zyl3et9nutje66645MnDjuQ7mXKAwAAAAAUGPXX3913nzzjQ/lXqIwAAAAAEBBRGEAAAAAKNy7776bceNG5/Of/0w+//ndc8EFZ2fu3LnN5++6644cccRXs8cen8zgwbvla187JL/97d3N53/5y7vy6U/vkscf/68cffThGTz4H3LAAfvlpptu/Lv3bGpqyoUXnpvBg3fLlCn3f6B5f//7e3PMMUdmzz0/lc98ZmAOOeSA3HrrT5a65rXXXssFF5yd/fbbM3vu+akce+w38l//9Vjz+SVLluT73788Bx30xQwevFu++tWD86tf/XypNe655zc54oivZs89B+ULX9gro0dfmDfffLP5/MiR5+TAA/df6jv/e5uMhx+emk9+cudMnTolw4YNzWc/u1u+8IW9ctll49PY2Jjkr1t4zJo1M7/61c/zyU/unJkzX/lAP48PShQGAAAAgML99rf/lqeempYRI87J0KEn5r77/pBTTjkhjY2NufXWn2T06AszaNCnM2rUuJx99vlp06ZNzj33jMye/WrzGtVqNWed9Z189rOfy+jRl2bbbbfPZZddmgceuG+Z9xw3bnTuvvtfc+GFozNgwK4rPOuf/vSHnH76yfn4xz+Riy76Xi64YFQ22GDDjB07Ko8//l9JknfeeSff+tYRefjhqTnmmONz4YWj0rZtuwwbdmxmzPhLkuTcc0fk5pt/kP33/2JGjRqbAQMGZuTIc/Jv//brJMl1103OOeecka233iYXXDAqhx9+VO69954cd9zRWbRo4Qf+GZ933pnZbrsdMmrUuOy551656aYbctdddyRJLrxwdLp3756BA3fLFVdcm+7dP/KB1/8gGlbr6gAAAADAGm/dddfLmDET06FDhyTJeuutl9NOOzn33/+nvPLKy/nnf/5qvva1I5uv79VrgxxxxKF57LE/Z4899kry1yd/Dz/8yOy335eSJP36bZd///ff5U9/+o/sssvApe53xRUT87Of3Z6RI0dn113/4QPNOn3689l77/1ywgnfbj7Wr9+22Wefz+bhh6dm6623ya9+dVdmzXol11zzg2yxxcf/55rtc/jhh+TPf344S5Yszr333pPjj/92Dj74n5MkO+88ILNmvZKHH34ou+zyD7nhhmvyhS8MyUknDW++z+ab98nQoUflF7+4K1/+8kEfaO799/9S889wp5365/e///f86U9/yJe+dEC23LJv2rRpm/XW65pttun3gdZdGaIwAAAAABRu4MDdmoNwkuy226dSX1+fRx99OMcdNyxJ8tZbb+XFF6fn5Zdn5OGHpyb56xYM/7+tt962+b/btm2b9dZbLwsWLFjqmttuuyXTpj2Rz39+3/zDP3zyA896yCH/kuSvTwP/5S8v5uWXZ2TatCf/Z57FSZLHHvtzevfeoDkIJ0n79u3zox/dliS5446fJkk+/enPLLX2yJGjkyT33ffHLF68uDl4v2e77XZIr16988gjD33gKPy/Y2+PHj2ycOGCv3P16iUKAwAAAEDhunXrvtTnSqWS9dZbL2+99VZefvmljBp1YR56aEratGmTTTbZNB/72BZJ/vp08P+vffv271vnf1/zzDNPZcCAgfm3f/t1Dj74n7Plln0/0Kzz5s3L6NEj8x//8e+pq6vLRhttnG233WGped5444107drt767xxhtvJMnfveatt/66b3D37t3fd65bt+6ZP/+tDzRzkrRrt/TPpq6uLtVq9QOv0xJEYQAAAAAo3HsR9D2NjY2ZN29e1l13vZxyyglp06ZNJk++IR/72JZpaGjICy88n3/911+u1L2OPPJbOfjgf8qhh/5jLrrognz/+9envr5+hb9/7rln5MUXp2fcuMuyzTbbpm3btlm4cGHuuuv25ms6deq8zJe1/ed/PprOnbukU6fOSZJ58+amR4+ezedffHF63nhjXjp37pIkef3117PJJpsutcbrr7+WDTbYMMmyw+6CBe+s8O+lVrxoDgAAAAAKN2XK/Xn33XebP9977z1pbGzMFlt8PH/5y4vZd98vpm/frdLQ8NdnTO+//09JslJPunbv3j3t2rXPSSedmqefnpYf//iHH+j7jz325+y+++DsuOPOadu27f/M88ckf3tSeLvtdsgrr7yc559/rvl7ixYtyhlnnJqf//zObLvt9kmSP/7xP5Za+/LLx+fSS7+XrbfeJm3bts3dd//rUucfffSRvPrqrObvd+y4TubNm5dFixYtNd/KqFQ+vFTrSWEAAAAAKNzrr7+WESNOzQEH/GNeemlGrrxyYvr33yWDB++RK66YkNtu+0l69OiRzp275IEH/pSf/ORHSbJKe+IOHLhbPvOZPXL11Vfm058enA033GiFvveJT2yd3/zm1/n4xz+R9dfvkf/8z0fzgx9cl7q6uub9i/fdd//89Kc35zvfOSlHHvnNrLvuernllh9lyZIl+fKXD8oGG2yYz3xmj1x22aVZuHBhtthiy9x//5/yxz/+R0aOHJ0uXdbNV75yWK67bnIaGhqy226fyiuvvJzJk6/Ipptunr333i9Jsttun8xPf3pzLr74guy33xfz3HPP5uabf/CBnnx+T6dOnfP000/lkUceylZbbf2+7SZakieFAQAAAKBwQ4YclK5du+X000/J979/efbcc+9ceOElqaury4UXXpKPfGT9jBx5bs466zt5/PH/ysUXj81HP7ppHn30z6t03xNO+HYaGhoyatSFK/ydESPOzVZbbZOxY0fl9NNPzh/+8O855ZTTM2DAwDz22CNJ/voE76RJ38/WW/fLmDGjctZZ30m1Ws2ECVc2b/1w1lnn58AD/ym33PKjnHrqsDz88IO54IKL86lP7Z4kOeKIo/Ptbw/PQw9NzfDhw3Lttd//n5A8ufmlfP3775qhQ0/Mo48+kpNPPj6//e1vcuGFl6xUFP7nfz40c+a8nm9/+7hMmzbtA3//g6hr+t87PdMiGhurmTPn7VqPAQAAAEDBlixZnNdfn5nu3XunTZu2tR6HVbS8P89u3dZJff3ynwO2fQQAAAAAUHP//57Gf0+lUvlQ995trURhAAAAAKCmZs58JQcd9IXlXnf44UfliCOO/hAmat1EYQAAAACgpj7ykfUzefINK3Qdq04UBgAAAABqqk2bNunbd6taj1EMG3AAAAAAABREFAYAAAAAKIgoDAAAAABQEFEYAAAAAKAgojAAAAAAQEEaaj0AAAAAALD2qlTqUqnU1eTe1WpTqtWmmtx7bSYKAwAAAAArpVKpy3pdO6a+UpsNCRqr1cyb+84Kh+ELLzw3d9/9m1x33U3ZZJOPLnXu9ddfy6GHHpyBA3fLWWedvzrGXWOIwgAAAADASqlU6lJfqeSmh/4zs+e//aHeu0endXLITv1SqdStcBQ+7riTMmXK/Rk1amQmTLgydXV/e8J5zJiL06FDhwwbdurqGnmNIQoDAAAAAKtk9vy38/Ibb9V6jOXq3LlzTjnl9AwfPiw/+9nt+eIXv5wkuffee/L739+bMWMmpHPnzjWecvUThQEAAACAYuy226Dstdfeufzy8Rk06NNp165dxo4dnSFDDkz//rtm+vQXMnHi2Dz66CPp2LFjdtyxf4499sR07/6RJMmMGX/J2LGj8/jjj6VabUq/fttm6NAT06fPx2r8O1txtdnsAwAAAACgRk444ZS0a9cul102PpMnX5mOHTvmmGNOyGuv/XeGDj0yG220SSZPvjEXXzwub789P9/85tezYMGCJMnZZ5+e9ddfP5Mn35irrroulUolp59+co1/Rx+MJ4UBAAAAgKJ06dIlJ598Wk4//ZS0adMmEyZclfbt2+fGG6/N+uv3zIkn/i3ynnfeRdl338/md7+7O/vss39eeeWl9O+/S3r33iANDQ057bSz8uKL01OtVlOp0Qv3PihRGAAAAAAozqBBu6dv30+kV68NsvXW2yRJnn56Wl544bnsueegpa5dvHhxpk9/IUly1FHHZPz47+X223+aHXbYMbvs8g/ZY4+91pognIjCAAAAAECh2rVrn/bt2zd/rlabsuOOO+fb3/7O+67t1OmvL6A74ICDM3jwHrnvvj/moYemZPLkK3L99ZNz7bU3pVu37h/a7Kti7cnXAAAAAACr0eab98mLL05Pjx49s9FGG2ejjTZOly5dMn789/L8889m7tw5GTPm4ixZsiT77LN/zjzz/Fx//Y/y+uuv55FHHq71+CtMFAYAAAAASDJkyIGZP39+zjtvRJ555uk888zTOeus0/Lkk09ks836pHPnLrnvvj/m4otH5plnnsrLL7+UO++8LW3atEnfvp+o9fgrzPYRAAAAAMAq6dFpnVZxzw022DATJ16ZK66YmGOOOSL19fXp12+7jB9/Rbp27ZokGT360kyaNC4nnHBMFi5cmC222DKjRo3Lhhtu1OLzrC51TU1NTbUeojVqbKxmzpy3az0GAAAAAAVbsmRxXn99Zrp37502bdq2+PqVSl3W69ox9TV6yVpjtZp5c99JtVpG4lzen2e3buukvn75fxaeFAYAAAAAVkq12pR5c99JpVJXs/uXEoRbkigMAAAAAKw0YXbt40VzAAAAAAAFEYUBAAAAAAoiCgMAAAAAFGSN2lP4yiuvzB/+8IfceOONzcd++9vfZtKkSXn++efTtWvX7LXXXjnhhBPSvn37JMmiRYty0UUX5de//nUWLlyYwYMH54wzzki3bt2a17jvvvsyevToPPfcc+ndu3eOO+647Lvvvs3nV2QN+LBUKnU125y9NbGfEQAAAMCyrTFR+Ic//GHGjRuXnXfeufnY1KlTc+yxx+b444/P5z//+bz44os566yzMm/evHz3u99NkpxzzjmZOnVqJkyYkLZt2+bss8/O8ccfnx/84AdJkueeey5HH310Dj/88IwePTr33ntvTj311HTr1i0DBw5coTXgw1Kp1GW99Tqmvt5D/KuqsbGaefPeEYYBAAAA/peaR+FXX301Z599dh544IFsuummS527+eabs8suu+Sb3/xmkmTTTTfNsGHDMmLEiJx77rmZO3du7rjjjlxxxRXNMXnMmDH5/Oc/n0ceeSQ77LBDrr/++nz84x/PsGHDkiR9+vTJE088kcmTJ2fgwIF59dVXl7sGfFgqlbrU11cy6pZfZcbsObUeZ621cY9uOfWgvVOp1InCAAAAAP9LzaPw448/njZt2uRnP/tZJk2alJdffrn53Ne//vVUKks/MVmpVLJkyZLMnz8/Dz30UJJk1113bT6/2WabpWfPnnnwwQezww47ZOrUqdljjz2WWmPXXXfNyJEj09TUtEJrwIdtxuw5eW7m7FqPAQAAAEArVPMoPHjw4AwePHiZ57baaqulPi9ZsiTXXXddttlmm3Tr1i2vvvpqunbtmnbt2i11XY8ePTJr1qwkyaxZs9KrV6/3nV+wYEHmzp27QmsAAAAAAMtWy/cjrcw7hQ48cP8kyQ033JyOHddZ6tzIkedk5sxXMnHiVS0245qo5lF4Rb377rs59dRT88wzz+SHP/xhkmTBggVp27bt+65t165dFi1alCRZuHDh+6557/PixYtXaA0AAAAA4P1q/X6klX2n0KxZMzNp0qU55ZTTV9Nka7a1IgrPnz8/J554YqZMmZKJEydm2223TZK0b98+ixcvft/1ixYtSocOHZL8Ne7+72ve+9yhQ4cVWgMAAAAAeL9avh9pVd4ptMEGG+bOO2/L7rt/Nv3777KaJlxzrfFRePbs2TnqqKPy8ssv5+qrr07//v2bz/Xq1Svz5s3L4sWLl3rad/bs2enZs2eSpHfv3pk9e/b71uzYsWM6d+68QmsAAAAAAH/f2vZ+pL322iePPfZoLrro/Nx444/ft41Ekrz55hv5/vevyB//+PvMmzcvH//4x3PUUcdkxx13rsHELas2z3WvoDfeeCOHHXZY5syZkx/+8IdLBeEk2WmnnVKtVptfFpckL7zwQl599dXma3feeedMmTJlqe/df//92XHHHVOpVFZoDQAAAACgdTnttDPz1ltvZcKEce8719jYmGHDjs1jjz2SM888L1dffWM23/xjOemkY/Pkk49/+MO2sDU6Cn/3u9/NjBkzMnr06HTr1i3//d//3fyrsbExPXv2zL777psRI0bkgQceyGOPPZaTTjopAwYMyPbbb58k+epXv5rHHnssl1xySZ577rlcc801+fWvf50jjzwySVZoDQAAAACgdenVq3eGDj0hd911e6ZMuX+pc1Om3J+nnnoyZ599QXbYYadsttnmOfnk07L55n1y00031mjilrPGbh/R2NiYX/7yl1myZEkOO+yw952/5557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mPGXjB07Oo8//liq1ab067dthg49MX36fKzGv7MVZ09hAAAAAKAoJ5xwStq1a5fLLhufyZOvTMeOHXPMMSfktdf+O0OHHpmNNtokkyffmIsvHpe3356fb37z61mwYEGS5OyzT8/666+fyZNvzFVXXZdKpZLTTz+5xr+jD8aTwgAAAABAUbp06ZKTTz4tp59+Stq0aZMJE65K+/btc+ON12b99XvmxBP/FnnPO++i7LvvZ/O7392dffbZP6+88lL6998lvXtvkIaGhpx22ll58cXpqVarqdRob+UPShQGAAAAAIozaNDu6dv3E+nVa4NsvfU2SZKnn56WF154LnvuOWipaxcvXpzp019Ikhx11DEZP/57uf32n2aHHXbMLrv8Q/bYY6+1JggnojAAAAAAUKh27dqnffv2zZ+r1absuOPO+fa3v/O+azt1+usL6A444OAMHrxH7rvvj3nooSmZPPmKXH/95Fx77U3p1q37hzb7qlh78jUAAAAAwGq0+eZ98uKL09OjR89stNHG2WijjdOlS5eMH/+9PP/8s5k7d07GjLk4S5YsyT777J8zzzw/11//o7z++ut55JGHaz3+ChOFAQAAAACSDBlyYObPn5/zzhuRZ555Os8883TOOuu0PPnkE9lssz7p3LlL7rvvj7n44pF55pmn8vLLL+XOO29LmzZt0rfvJ2o9/gqzfQQAAAAAsEp6dFqnVdxzgw02zMSJV+aKKybmmGOOSH19ffr12y7jx1+Rrl27JklGj740kyaNywknHJOFCxdmiy22zKhR47Lhhhu1+DyriygMAAAAAKyUarUpjdVqDtmpX03u31itplptWunvT5x41fuObbll34wZM/HvfmfTTTfL6NGXrvQ91wSiMAAAAACwUqrVpsyb+04qlbqa3X9VonCpRGEAAAAAYKUJs2sfL5oDAAAAACiIKAwAAAAAUBBRGAAAAACgIKIwAAAAAEBBRGEAAAAAgIKIwgAAAAAABRGFAQAAAAAK0lDrAQAAAACAtVelUpdKpa4m965Wm1KtNtXk3mszURgAAAAAWCmVSl3WW69j6utrsyFBY2M18+a984HC8IEH7p8kueGGm9Ox4zpLnRs58pzMnPlKJk68qkXnXNOIwgAAAADASqlU6lJfX8moW36VGbPnfKj33rhHt5x60N6pVOo+8NPCs2bNzKRJl+aUU05f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AAVZK6Lwu+++m0svvTSf+cxnssMOO+QrX/lK/vznPzeff/LJJ3PooYdm++23z+DBg3PDDTcs9f1qtZrx48dn0KBB2X777XPUUUdlxowZS12zvDUAAAAAAFqDtSIKX3755bnlllty/vnn54477shmm22WI488MrNnz87cuXNz+OGHZ5NNNsmtt96aoUOH5pJLLsmtt97a/P3LLrssN910U84///zcfPPNqVarOfLII7N48eIkWaE1AAAAAABag4ZaD7Ai7r777uy333755Cc/mST5zne+k1tuuSV//vOf88ILL6RNmzY577zz0tDQkD59+uTFF1/MVVddlQMOOCCLFy/ONddck5NPPjm77757kmTs2LEZNGhQfvOb32S//fbLT37yk/9zDQAAAACA1mKteFK4e/fu+d3vfpeXXnopjY2N+fGPf5y2bdumb9++mTp1agYMGJCGhr/17V133TXTp0/Pa6+9lmnTpuXtt9/OwIEDm8936dIlW221VR588MEkWe4aAAAAAACtxVoRhc8444y0adMmn/3sZ9OvX7+MHTs248ePzyabbJJZs2alV69eS13fo0ePJMnMmTMza9asJEnv3r3fd81755a3BgAAAABAa7FWbB/x7LPPpnPnzpk0aVJ69uyZW265JSeffHJ+8IMfZOHChWnbtu1S17dr1y5JsmjRoixYsCBJlnnNG2+8kSTLXQMAAAAAoLVY46PwzJkz8+1vfzvXXXdddt555yRJv3798uyzz2bChAlp37598wvj3vNeyO3YsWPat2+fJFm8eHHzf793TYcOHZJkuWsAAAAAALQWa/z2EY8++miWLFmSfv36LXV8u+22y4svvphevXpl9uzZS51773PPnj2bt41Y1jU9e/ZMkuWuAQAAAADQWqzxUfi9vX6feuqppY4//fTT2XTTTdO/f/889NBDaWxsbD53//33Z7PNNkv37t3Tt2/fdOrUKQ888EDz+TfffDNPPPFE+vfvnyTLXQMAAAAAoLVY46Pwtttum5122inDhw/P/fffn+nTp2fcuHG577778o1vfCMHHHBA5s+fnzPOOCPPPvtsbrvttlx33XU5+uijk/x1L+FDDz00l1xySe65555MmzYtw4YNS69evfK5z30uSZa7BgAAAABAa7HG7ylcqVRy+eWXZ9y4cTnttNPyxhtvZMstt8x1112X7bbb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AAAAWAOtVBS+5557WnoOAAAAAAA+BCsVhTfccMOWngMAAAAAgA/BSkXhiRMnLveaY489dmWWBgAAAABgNWrxKNypU6f06NFDFAYAAAAAWAOtVBSeNm3a+4698847mTp1as4555yceeaZqzwYAAAAAAAtr9JSC3Xs2DGf+tSnMnTo0IwaNaqllgUAAAAAoAW1WBR+zwYbbJDnnnuupZcFAAAAAKAFrNT2EcvS1NSUWbNmZfLkydlwww1balkAAAAAAFrQSkXhvn37pq6ubpnnmpqabB8BAAAAALCGWqkoPHTo0GVG4U6dOmX33XfPpptuuqpzAQAAAACwGqxUFD7uuONaeg4AAAAAAD4EK72n8Jw5c3LNNddkypQpefPNN9O1a9fsvPPO+drXvpbu3bu35IwAAAAAALSQysp8adasWRkyZEiuv/76tGvXLltttVUaGhpy7bXX5ktf+lJeffXVlp4TAAAAAIAWsFJPCo8ePToNDQ355S9/mY033rj5+IwZM/L1r389Y8eOzUUXXdRiQwIAAAAA0DJW6knhP/zhDzn++OOXCsJJsvHGG2fo0KH5/e9/3yLDAQAAAADQslYqCjc2NqZr167LPNetW7fMnz9/lYYCAAAAAGD1WKko/PGPfzx33XXXMs/deeed2XLLLVdpKAAAAAAAVo+V2lP4mGOOyRFHHJE33ngj++yzT9Zff/3893//d37xi1/kD3/4Q8aPH9/ScwIAAAAA0AJWKgrvtttuueiii3LJJZcstX/w+uuvn+9+97vZc889W2xAAAAAAABazkpF4SSZPXt2ttpqqwwfPjxvvPFGpk2blgkTJthPGAAAAABgDbZSUfiaa67JuHHjcuihh6ZPnz5Jkt69e+f555/PRRddlHbt2uWggw5q0UEBAAAAAFh1KxWFb7755px44on5xje+0Xysd+/eGTFiRD7ykY/kuuuuE4UBAAAAANZAlZX50quvvpp+/fot89x2222Xl156aZWGAgAAAABg9VipKLzhhhvmvvvuW+a5Bx98ML169VqloQAAAAAAWD1WavuIgw8+OKNHj86SJUuyxx57pHv37pkzZ05+97vf5dprr823v/3tlp4TAAAAAIAWsFJR+Gtf+1peffXV3Hjjjbnuuuuaj9fX1+ewww7L4Ycf3lLzAQAAAADQglYqCifJ8OHDc8wxx+TPf/5z5s2bly5dumTbbbdN165dW3I+AAAAAABa0EpH4STp3LlzBg0a1FKzAAAAAACwmq3Ui+YAAAAAAFg7icIAAAAAAAVZa6LwHXfckX322Sf9+vXLvvvum1/96lfN51566aUcffTR2XHHHfPJT34y48aNS2Nj41Lf/+EPf5jPfvaz2XbbbXPIIYfkiSeeWOr8iqwBAAAAALC2Wyui8J133pkzzjgjX/nKV/KLX/wi++23X0466aQ88sgjWbJkSY444ogkyc0335xzzjknP/rRjzJp0qTm799+++0ZNWpUTjjhhNx2223ZaKONcvjhh2fOnDlJskJrAAAAAAC0Bqv0orkPQ1NTUy699NL8y7/8S77yla8kSb71rW9l6tSpmTJlSl5++eW88sor+clPfpJ11103W265ZV5//fWMGjUq3/zmN9O2bdtcccUVOfTQQ/OFL3whSXLhhRdmjz32yC233JKjjz46//qv/7rcNQAAAAAAWoM1/knhF154IS+//HL233//pY5fffXVOfroozN16tRsvfXWWXfddZvP7brrrpk/f36efPLJvP7665k+fXoGDhzYfL6hoSE777xzHnzwwSRZ7hoAAAAAAK3FWhGFk+Sdd97JEUcckYEDB+aggw7Kb3/72yTJrFmz0qtXr6W+06NHjyTJzJkzM2vWrCRJ796933fNe+eWtwYAAAAAQGuxxkfh+fPnJ0mGDx+e/fbbL9dcc0122223HHPMMbnvvvuycOHC923v0K5duyTJokWLsmDBgiRZ5jWLFi1KkuWuAQAAAADQWqzxewq3adMmSXLEEUdkyJAhSZJPfOITeeKJJ3Lttdemffv2Wbx48VLfeS/kduzYMe3bt0+SZV7ToUOHJFnuGgAAAAAArcUa/6Rwz549kyRbbrnlUsc/9rGP5aWXXkqvXr0ye/bspc6997lnz57N20Ys65r31l7eGgAAAAAArcUaH4W33nrrrLPOOnn00UeXOv70009nk002Sf/+/fPEE080bzORJPfff3/WWWed9O3bN927d89mm22WBx54oPn8u+++m6lTp6Z///5Jstw1AAAAAABaizU+Crdv3z5HHnlkJk2alJ///Of5y1/+kssvvzx//OMfc/jhh2ePPfbI+uuvnxNPPDHTpk3L3XffnTFjxuTrX/968z7BX//613Pttdfm9ttvz7PPPpvTTz89CxcuzIEH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+clP0qZNm5x33nlpaGhInz598uKLL+aqq67KAQcckMWLF+eaa67JySefnN133z1JMnbs2AwaNCi/+c1vst9++y13DQAAAACA1mSteFL4wQcfzI9//ONcdNFFSx2fOnVqBgwYkIaGv7XtXXfdNdOnT89rr72WadOm5e23387AgQObz3fp0iVbbbVVHnzwwRVaAwAAAACgNVnjo/Cbb76ZU089NSNGjEjv3r2XOjdr1qz06tVrqWM9evRIksycOTOzZs1Kkvd9r0ePHs3nlrcGAAAAAEBrssZH4XPOOSc77LBD9t9///edW7hwYdq2bbvUsXbt2iVJFi1alAULFiTJMq9ZtGjRCq0BAAAAANCarNF7Ct9xxx2ZOnVq7rrrrmWeb9++ffML497zXsjt2LFj2rdvnyRZvHhx83+/d02HDh1WaA0AAAAAgNZkjY7Ct956a15//fXml8S95+yzz84vf/nL9OrVK7Nnz17q3Hufe/bsmXfffbf52CabbLLUNR//+MeTZLlrAAAAAAC0Jmt0FL7kkkuycOHCpY597nOfy/HHH58vfOELufPOO3PzzTensbEx9fX1SZL7778/m222Wbp3757OnTunU6dOeeCBB5qj8Jtvvpknnngihx56aJKkf//+/+caAAAAAACtyRq9p3DPnj3z0Y9+dKlfSdK9e/f07NkzBxxwQObPn58zzjgjzz77bG677bZcd911Ofroo5P8dS/hQw89NJdccknuueeeTJs2LcOGDUuvXr3yuc99LkmWuwYAAAAAQGuyRj8pvDzdu3fP5MmTM3LkyAwZMiTrr79+Tj311AwZMqT5muOPPz7vvvtuRowYkYULF6Z///65+uqr06ZNmxVeAwAAAACgtVjrovBTTz211Odtt902P/7xj//u9fX19TnllFNyyimn/N1rlrcGAAAAAEBrsUZvHwEAAAAAQMsShQEAAAAACiIKAwAAAAAURBQGAAAAACiIKAwAAAAAUBBRGAAAAACgIKIwAAAAAEBBRGEAAAAAgIKIwgAAAAAABRGFAQAAAAAKIgoDAAAAABREFAYAAAAAKIgoDAAAAABQEFEYAAAAAKAgojAAAAAAQEFEYQAAAACAgojCAAAAAAAFEYUBAAAAAAoiCgMAAAAAFEQUBgAAAAAoiCgMAAAAAFAQURgAAAAAoCCiMAAAAABAQURhAAAAAICC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5s9f8Inb2rUrpaamXd6b/lraL176iXNqOnRIp64b/dVzL0m7zF+v88oXDQAAAG1Qqw+Fu3TpkkGDBjUbmzRpUmbMmJGLLroov/zlL9OjR49m2zfbbLMsWrQoc+bMydtvv51u3bpl/fXXX2HOrFmzkiSzZs36m8fYaKO/HiQAsPreeGNG5s2bnZqaT17VaJtttvrUY9TW1q7y+RsbG/NvT72YxTUdVvkYAAAA0Fa0+lD4Lz3zzDO58MILc8ABB2Tw4MFZvHhx2rdv32zOx79funRpFi1atML2JFl//fWzZMmSJPnUYwCwZvXsuVXmzJm7Wk8K/2H6a1m4Gk8KL/akMAAAAAXRpkLhhx9+OOeff3769u2b+vr6JB+Fu38Z3H78+44dO6ZDhw6fGOwuWbIkHTt2/EzHAGDN691711Xar2PH9dK5c4d8e+zPMuethk+cszTJB6tRGwAAAKxLPvk93Vbopz/9ab797W9nyJAhGT9+fNNyEJtvvnkaGpqHAA0NDenUqVM23HDD9OjRI3Pnzl0h9G1oaEj37t0/0zEAAAAAANYVbSIUnjhxYi655JIce+yxGT16dLOlHnbffff87ne/azb/ySefTN++fVMul9OvX780NjY2feFckrzyyit5++23079//890DAAAAACAdUWrTzxfeeWVXH755dl///1z2mmn5Z133sns2bMze/bszJ8/P8cff3yeffbZ1NfXZ/r06bn55pvz4IMP5pRTTkmSdO/ePYccckhGjBiRp556Ks8++2zOPffc7LHHHtltt92S5FOPAQAAAACwrmj1awpPmjQpH374YSZPnpzJkyc323bEEUfkyiuvzLhx4zJq1Kjceuut6dmzZ0aNGpUBAwY0zbvkkkty+eWXZ9iwYUmSffbZJyNGjGjavv3223/qMQAAAAAA1gWtPhQ+/fTTc/rpp//NOfvss0/22Wefv7q9U6dOufTSS3PppZeu8jEAAAAAANYFrX75CAAAAAAAWo5QGAAAAACgQITCAAAAAAAFIhQGAAAAACgQoTAAAAAAQIEIhQEAAAAACkQoDAAAAABQIEJhAAAAAIACqal2AawbyuVSyuVStcto83yGAAAAAKxpQmFWW7lcStdundKu7MFzAAAAAGjthMKstnK5lHblciY+PS0NCxZWu5w2rf8XPp8vb7tltcsAAOATlEre6mppPlMA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uOOOS319fTbaaKNsscUWGTVqVHr06JEDDjig2uUBAAAAALQYofCfDB8+PMuWLcuIESOyePHi9O/fPz/+8Y+z3nrrVbs0AAAAAIAWU6pUWvN3vQIAAAAA0JKsKQwAAAAAUCBCYQAAAACAAhEKAwAAAAAUiFAYAAAAAKBAhMIAAAAAAAUiFAYAAAAAKBChMAAAAABAgQiFAQAAAAAKRCgMAAAAAFAgQmEAAAAAgAIRCsMacPnll2e//fZrNjZ//vzU1dXl17/+dZ555pkce+yxqaury+DBgzNy5MgsWLCgae6bb76Zc845JwMGDMguu+ySffbZJ6NGjUpjY2OS5K677sr++++fSy+9NP369cuZZ565Vq8P1pS///u/z84779xs7K233kptbW3GjBmT22+/Pf369UttbW122mmnDB06NLNmzWqaO3Xq1AwePDg77rhj05yjjz46y5YtS5J8//vfzy677JLDDz88tbW1+fKXv7xWrw/WBPccWDV6B1aevoFVo3dojYTCsAYceeSRef311zNlypSmsfvvvz9dunRJjx49cuKJJ2bgwIG55557Ul9fn+eeey4nnXRSKpVKkuSMM87I/Pnzc8stt+TBBx/MSSedlAkTJuSRRx5pOt5rr72WhoaG3H333TnnnHPW+jXCmnD66adn+fLl+dnPftY0Nnr06JTL5Wy33Xb553/+5+y66665+eabM2LEiMycOTMHHXRQ0x+GTjjhhCxevDjXXHNNbrvttgwZMiTPPvtsrrvuuqbjLVu2LHPmzMltt92WkSNHrvVrhJbmngOrRu/AytM3sGr0Dq1SBVgjjjjiiMrFF1/c9PtjjjmmctVVV1XOP//8yhlnnNFs7muvvVbZYYcdKk8++WRl0aJFlR//+MeVN998s9mcvfbaq3LDDTdUKpVK5c4776zssMMOlRdeeGHNXwisZXV1dZUDDzyw6fd9+vSpHHXUUZV99923stdeezWb+9RTT1V22GGHyi233FKZM2dO5bTTTqv893//d7M5tbW1lW984xuVSqVSueiiiyo77LBD5YEHHljzFwJrkXsOrBq9AytP38Cq0Tu0NjXVDqVhXXXUUUfl2muvzYgRI/LWW29l6tSpueyyyzJ8+PDMmDEjffr0WWGf6dOnZ88998xxxx2XBx98MM8++2xmzJiRP/7xj3nnnXeanob82NZbb72WrgbWniFDhuSBBx7IggUL8uyzz2bhwoU566yzMnz48CxdujS1tbUr7POHP/wh3/zmN1NfX5/6+vpcfPHFaWhoyLx581KpVLJ8+fJm87/0pS+trcuBtcI9B1aN3oGVp29g1egdWhuhMKwhhx56aK666qo8+uijeemll1JXV5devXqlsbExhx56aE4//fQV9tloo43ywQcf5LjjjsvixYtz4IEH5ogjjkhdXV2OPfbYFeZ36NBhbVwKrFXf+c538sADD2Ts2LGZNm1aOnbsmCFDhqRSqWTLLbf8xCUfttlmm7z77rvZd999s3z58uy6667Zc88983d/93c5/vjjV5jftWvXtXAlsPa458Cq0Tuw8vQNrBq9Q2sjFIY1pEuXLtl///0zefLkvPjii00/sLfffvv83//9X7baaqumudOnT8+oUaNy7rnn5tVXX81zzz2Xxx9/PJtsskmSZO7cuXn33Xeb1hOCddkWW2yRz3/+85k0aVJmz56d/fffP8lHfyCaPXt29tprr6a5v/71r3PRRRdl5MiRmTp1ahYvXpx77703O+ywQ5JkxowZaWxs1Dus89xzYNXoHVh5+gZWjd6htfFFc7AGHXXUUZk8eXJee+21HHLIIUmSk046Kc8//3xGjhyZ6dOnZ+rUqTnvvPPy6quvZuutt06PHj2SJPfcc09mzpyZKVOm5Mwzz8yHH36YpUuXVvNyYK35+te/npkzZ2bp0qU577zzkiTDhg3LokWLMnTo0Dz66KP5+c9/nuHDh+f999/PXnvtlV69eiX56Ivpnn766fzsZz/LkUcemST58MMPq3YtsLa458Cq0Tuw8vQNrBq9Q2viSWFYgwYMGJBu3bqlb9++6dKlS5Jkt912y4QJE3LdddfliCOOSKdOnTJgwIB897vfTfv27VNXV5cLL7wwP/nJT3Lttdeme/fuOfjgg7P55ptn2rRpVb4iWDtOPvnkjB49Oj169MgWW2yRJPnHf/zHzJ8/P2PHjs3pp5+eUqmUHj165Prrr88GG2yQo446Kv/xH/+RRx99NI8++mhqampSV1eXWbNm5ZVXXqnyFcGa554Dq0bvwMrTN7Bq9A6tSaniWXNYYxYuXJi99947Y8eObfbKO/C3zZ49O3vvvXfOP//8nHrqqdUuB9oE9xxYNXoHVp6+gVWjd2hNPCkMa8C8efPy5JNP5oEHHsgWW2yRAQMGVLskaBNef/31TJw4Mffff3/at2+fk08+udolQavnngOrRu/AytM3sGr0Dq2RUBjWgOXLl+f73/9+Ntpoo1x77bUplUrVLgnahKVLl+bmm29OTU1NrrrqqpTLlr6HT+OeA6tG78DK0zewavQOrZHlIwAAAAAACsQjWAAAAAAABSIUBgAAAAAoEKEwAAAAAECBCIUBAAAAAApEKAwAAGuJ73gGAKA1EAoDAFAIb7zxRmpra3PXXXet9XPPmjUr3/rWtzJz5symsX333Tff+9731notSTJmzJjU1tZW5dy1tbUZM2ZMVc4NAMBHaqpdAAAArOv+67/+K4899lizsRtuuCGdO3euUkUAABSZUBgAAKpg5513rnYJAAAUlOUjAABok37xi1/kkEMOyRe/+MUMHjw4Y8aMyfLly5u2P/TQQznssMNSV1eXI444Ii+++GKz/e+6667U1tbmjTfeaDb+l8s6LF26NNdee22+8pWvpK6uLkOHDs0vf/nLpu3Lly/PTTfdlKFDh6auri677bZbvvrVr+bJJ59sOs+FF16YJPnKV77SdOy/PM/8+fNzxRVXZL/99kvv3r0zdOjQ3HHHHSvUdv311+eqq67KXnvtlbq6upx88sl59dVXV+OT/MjDDz+cI488Mr17986Xv/zlXHrppfnggw+SJM8880xqa2vz6KOPNtvnhRdeSG1tbSZPnpwkWbJkSX74wx9m0KBB+eIXv5hDDz00999//2rXBgBAyxIKAwDQ5tx44425+OKLM2DAgIwfPz7HHntsfvSjH+Xiiy9OkjzyyCMZPnx4amtrM3bs2Bx00EH5zne+s0rnOv/883PLLbfk6KOPzo033pi999473/ve9/KrX/0qSVJfX59x48blmGOOyYQJE3LJJZdk7ty5Ofvss7No0aIMHjw4Z5xxRpKPlow488wzVzjH4sWL8/Wvfz333ntvTjnllIwbNy79+vXL97///YwfP77Z3Ntuuy0vv/xyrrjiilx66aX5n//5n3z3u99dpWv72L333puzzjor2267bcaOHZthw4blnnvuyZlnnplKpZK+fftmyy23zH333ddsv1/96lfp2rVrBg0alEqlkrPOOiu33357TjzxxPzrv/5r+vTpk3POOSd33333atUHAEDLsnwEAABtyvz585tC2BEjRiRJ9t5773Tt2jUjRozIiSeemLFjx6auri6jRo1KkgwcODBJcvXVV6/UuV566aVMmjQpF110UU444YQkyYABAzJz5sw89dRTGTp0aBoaGnLOOefk+OOPb9pv/fXXz7e//e388Y9/zG677ZYtt9wySbLTTjulZ8+eK5znrrvuyksvvZTbb789ffr0aap52bJlGTduXL761a+ma9euSZIuXbpk3LhxadeuXZLktddey5gxYzJnzpx069Ztpa4vSSqVSurr6zNw4MDU19c3jW+99db55je/mcceeyyDBw/OYYcdlptvvjmLFy9Ohw4dUqlUcv/99+fAAw9M+/bt8/jjj+e3v/1trrnmmhx88MFN17Bo0aLU19dn6NChqanxvx8AAK2BJ4UBAGhTpk6dmsWLF2fffffNsmXLmn7tu+++ST56Svi5557LkCFDmu130EEHrfS5nn766STJAQcc0Gx8zJgxueSSS5J8FDSfcMIJee+99zJlypTceeedueeee5J8tPTEZ/G73/0uW2yxRVMg/LHDDjssS5YsyR/+8Iemsd69ezcFwknSo0ePJMmiRYtW8uo+8vLLL2fWrFkrfJ79+/dP586d8/jjjzfV8sEHHzQtIfHMM8/kzTffzOGHH54keeKJJ1IqlTJo0KAV/r3Mnj07//u//7tK9QEA0PL8VT0AAG3K3LlzkyTf+ta3PnH7G2+8kUqlssJTs5ttttkqn2vjjTf+q3OmTZuWkSNHZtq0aenYsWO22267fP7zn0/y0VO4n8W8efOy6aabrjC+ySabJEnef//9prGOHTs2m1Muf/ScR2Nj42c611/6+BpHjhyZkSNHrrC9oaEhSbLVVlulT58+ue+++3LQQQflvvvuy5Zbbpm+ffs2HefjpSY+SUNDQ3baaadVqhEAgJYlFAYAoE3p0qVLko/W8t16661X2L7JJpvkjjvuyDvvvNNs/OPw82OlUinJimHqwoULVzjXe++91/REbpJMnz49c+fOTW1tbU455ZTU1tbmvvvuy7bbbptyuZzHHnsskyZN+szX9LnPfS4zZsxYYXz27NlJskrLQnxWH1/jBRdckD322OMTa/vYYYcdliuuuCLz58/Pgw8+mK997WtN2zbccMN06tQpt9122yeeZ6uttmrhygEAWFWWjwAAoE3Zdddds9566+Xtt99O7969m37V1NRk9OjReeONN9KnT5889NBDzZ7UfeSRR5odp3PnzkmSWbNmNY19HPZ+rF+/fp+4b319fS677LK8/PLLmTt3br7xjW9ku+22a3pq9ze/+U2SPwfOH4//Nf3798/MmTMzderUZuP33HNP1ltvvdTV1X3q57Kqtt1222y88cZ54403mn2e3bt3z9VXX53nn3++ae7BBx+cSqWS6667Lu+++24OO+ywpm177LFHPvjgg1QqlWbHeemllzJ27NgsW7ZsjV0DAAArx5PCAAC0Kd26dcspp5yS6667LgsWLMiee+6Zt99+O9ddd11KpVJ23HHHnHvuuTnhhBMybNiwHHPMMXnllVcyfvz4ZsfZc88906FDh1x55ZU5++yzs3Dhwlx//fVNX+iWJDvuuGMOPPDAjBo1KosXL85OO+2U3/zmN3n00Udzww03ZJtttknnzp0zfvz41NTUpKamJpMmTcodd9yR5M/r/H78NO7kyZOzzz77pFevXs1qOfLIIzNx4sScddZZGT58eHr27JlHHnkkd955Z4YNG9a0/5rQrl27nHPOOfnBD36Qdu3aZciQIXn//fczbty4vP3229lll12a5nbt2jWDBg3KxIkT06dPn2ZP/w4aNCj9+/fPmWeemTPPPDO9evXKs88+m+uvvz4DBw7MRhtttMauAQCAlSMUBgCgzfmnf/qnbLrpppk4cWImTJiQz33ucxkwYEDOPffcbLjhhtl9993zox/9KKNHj86wYcPSs2fPXH755Tn99NObjtGlS5eMGTMmV199dc4666xsscUWGTZsWO6+++5m5xo1alRuuOGG3HrrrZkzZ0569eqV66+/Pvvtt1+SZNy4cfnhD3+Ys88+OxtssEF22mmn/PSnP82pp56aKVOmZN99982ee+6ZvfbaK1dffXWeeOKJ3HTTTc3O0bFjx/zbv/1brr766qawe9ttt81ll12Wf/iHf1jjn+fRRx+dDTbYIBMmTMjPf/7zdOrUKX379k19fX2+8IUvNJt7+OGH5+GHH86hhx7abLxcLuemm27KddddlxtvvDHvvvtuunfvnhNPPDFnnXXWGr8GAAA+u1Lls377BQAAAAAAbZ4nhQEAYB3xWdbtLZfLn7rGMQAA6zZPCgMAwDqitrb2U+ccccQRufLKK9dCNQAAtFZCYQAAWEdMmzbtU+d069YtPXv2XAvVAADQWgmFAQAAAAAKxGJiAAAAAAAFIhQGAAAAACgQoTAAAAAAQIEIhQEAAAAACkQoDAAAAABQIEJhAAAAAIACEQoDAAAAABSIUBgAAAAAoED+H0HE3cYVQTWdAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1700x1500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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b7wRM2fOjB//+MdN77nkkkti+fLlcffdd7f6mcccc0wcffTRcemll0ZExIYNG+Loo4+Ok08+Oc4666wt+kcVCo3R0LB6i94LAEDnM3nypKitvTLq6+c3rdXUDIra2mvjuOM+VcbJAADYUVRX94qqqrbPv7brhGy/fv3izjvvjPe///1NaxUVFVFRURErVqyIGTNmxGGHHdbsPaNHj44//vGP0Vr3feONN2LevHnN3tOlS5cYOXJkTJ8+vT2jAQDAFpk8eVJMnDg+hg0bHlOmTI05cxbElClTY9iw4TFx4viYPHlSuUcEAKADaVeQ7dOnTxx55JHRrVu3prXHH3885s+fHx/5yEdi0aJFMXDgwGbv6d+/f7z55puxbNmyFp+3aNGiiIjYfffdW7znnWcAALCtFAqFqK29MsaO/UTcd98jMXLkobHzzjvHyJGHxn33PRJjx34iamuvcn0BAAAl0+47ZDf27LPPxhVXXBFjx46No446KtauXdss1kZE0+v169e3eP+bb77ZbM87unfvHuvWrdua0QAAoE11dU9Fff38uOCCS6KysvmvxpWVlXH++RdHff28qKt7qkwTAgDQ0WxxkJ06dWp8/vOfj4MPPjhuvPHGiHg7pP5jeH3n9U477dTiM3r06NFszzvWrVvX6n4AACilxYvf/qus/fcf3urzYcOGN9sHAABba4uC7IMPPhjnnXdefPSjH43bb789unfvHhFvXz2wZMmSZnuXLFkSPXv2jN69e7f4nHeuKmjtPQMGDNiS0QAAYLMNGPD2dVsvvvhCq89nznyh2T4AANha7Q6yDz/8cHzjG9+IU045JW666aZm1w2MHDkynnnmmWb76+rqYsSIES3+BCwiYpdddokhQ4bEtGnTmtY2bNgQM2bMiEMOOaS9owEAQLuMHn141NQMiltu+XY0NjY2e9bY2Bi33npT1NQMjtGjDy/ThAAAdDTtCrJz586N6667Lj7+8Y/HWWedFa+//nosXbo0li5dGitXrozx48fH888/HzfeeGO88sor8cMf/jAee+yx+MIXvtD0GcuXL4/ly5c3vf785z8f99xzT/z0pz+Nl19+Ob761a/G2rVr46STTirZPxIAAFpTVVUVtbXXxhNPPBYTJpwc06dPi1WrVsb06dNiwoST44knHova2m9GVVVVuUcFAKCDqCgWi8XN3Xz77bfHd77znVafjRs3Lv7t3/4tfv/738e///u/x7x582LPPfeM8847L4499timfePHj4+IiAceeKBp7e677477778/li9fHgcccEBcddVVMWzYsC39N0Wh0BgNDau3+P0AAHQukydPitraK6O+fn7TWk3N4Kit/WYcd9ynyjgZAAA7iurqXlFV1fb513YF2R2FIAsAQHsVCoWoq3sqFi9eFAMGDIzRow93MhYAgM0myAqyAAAAAECSzQ2y7f5SLwAAAAAAtowgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSdCn3AAAAsD0oFApRV/dULF68KAYMGBijRx8eVVVV5R4LAIAORpAFAKDTmzx5UtTWXhn19fOb1mpqBkVt7bVx3HGfKuNkAAB0NK4sAACgU5s8eVJMnDg+hg0bHlOmTI05cxbElClTY9iw4TFx4viYPHlSuUcEAKADqSgWi8VyD1FqhUJjNDSsLvcYAABs5wqFQowadXAMGzY87rvvkais/N/zCo2NjTFhwskxc+bMmDbtT64vAADgXVVX94qqqrbPvzohCwBAp1VX91TU18+PCy64pFmMjYiorKyM88+/OOrr50Vd3VNlmhAAgI5GkAUAoNNavHhRRETsv//wVp8PGza82T4AANhagiwAAJ3WgAEDIyLixRdfaPX5zJkvNNsHAABbS5AFAKDTGj368KipGRS33PLtaGxsbPassbExbr31pqipGRyjRx9epgkBAOhoBFkAADqtqqqqqK29Np544rGYMOHkmD59WqxatTKmT58WEyacHE888VjU1n7TF3oBAFAyFcVisVjuIUqtUGiMhobV5R4DAIAdxOTJk6K29sqor5/ftFZTMzhqa78Zxx33qTJOBgDAjqK6uldUVbV9/lWQBQCAiCgUClFX91QsXrwoBgwYGKNHH+5kLAAAm02QFWQBAAAAgCSbG2TdIQsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCRdtubNd9xxRzz55JPxwAMPRETE+PHj45lnnml177e+9a04/vjjW312xhlnxFNPPdVs7dBDD236XAAAAACAjmCLg+xDDz0UN998c4wcObJp7bvf/W689dZbTa+LxWJcdNFF8fe//z0+/vGPb/KzZs2aFbW1tXH00Uc3rXXt2nVLRwMAAAAA2C61O8guXrw4vv71r8e0adNi8ODBzZ717du32esHH3wwnn/++fj5z38evXr1avXz3njjjXjjjTfioIMOit1226294wAAAAAA7DDafYfsX//61+jatWtMmjQpDjrooE3ua2hoiJtvvjnOOeec2GuvvTa5b9asWVFRURFDhgxp7ygAAAAAADuUdp+QHTNmTIwZM6bNfXfddVf06NEjJk6c+K77Zs+eHb17945rrrkm/vCHP0TPnj3jE5/4RJx77rnRrVu39o4HAAAAALDd2qov9dqUVatWxY9+9KP48pe/HN27d3/XvbNnz45169bFgQceGGeccUbMnDkzbrjhhliwYEHccMMN22I8AAAAAICy2CZBdurUqbF+/fo48cQT29x7zTXXxOWXXx7vec97IiJi3333ja5du8ZFF10Ul112Wey6667bYkQAAAAAgHTtvkN2c0ydOjWOPPLI6NOnT5t7u3Tp0hRj37HPPvtERMSiRYu2xXgAAAAAAGWxTYLsjBkz4rDDDtusvePHj48rrrii2dqf//zn6Nq1awwePHgbTAcAAAAAUB4lD7ILFy6MZcuWxf7779/q89WrV8fSpUubXh9zzDHx85//PB555JF49dVXY8qUKXHDDTfExIkTY+eddy71eAAAAAAAZVPyO2Tfia19+/Zt9fkPf/jDuO2222LWrFkREXHqqadGRUVFPPDAA3HdddfFbrvtFqeffnqceeaZpR4NAAAAAKCsKorFYrHcQ5RaodAYDQ2ryz0GAAAAANBJVFf3iqqqti8k2CZ3yAIAAAAA0JIgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEm6lHsAAADYHhQKhaireyoWL14UAwYMjNGjD4+qqqpyjwUAQAcjyAIA0OlNnjwpamuvjPr6+U1rNTWDorb22jjuuE+VcTIAADoaVxYAANCpTZ48KSZOHB/Dhg2PKVOmxpw5C2LKlKkxbNjwmDhxfEyePKncIwIA0IFUFIvFYrmHKLVCoTEaGlaXewwAALZzhUIhRo06OIYNGx733fdIVFb+73mFxsbGmDDh5Jg5c2ZMm/Yn1xcAAPCuqqt7RVVV2+dfnZAFAKDTqqt7Kurr58cFF1zSLMZGRFRWVsb5518c9fXzoq7uqTJNCABARyPIAgDQaS1evCgiIvbff3irz4cNG95sHwAAbC1BFgCATmvAgIEREfHiiy+0+nzmzBea7QMAgK0lyAIA0GmNHn141NQMiltu+XY0NjY2e9bY2Bi33npT1NQMjtGjDy/ThAAAdDSCLAAAnVZVVVXU1l4bTzzxWEyYcHJMnz4tVq1aGdOnT4sJE06OJ554LGprv+kLvQAAKJmKYrFYLPcQpVYoNEZDw+pyjwEAwA5i8uRJUVt7ZdTXz29aq6kZHLW134zjjvtUGScDAGBHUV3dK6qq2j7/KsgCAEBEFAqFqKt7KhYvXhQDBgyM0aMPdzIWAIDNJsgKsgAAAABAks0Nsu6QBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAEBErFq1KiZMODmOPHJ0TJhwcqxatarcIwEA0AFVFIvFYrmHKLVCoTEaGlaXewwAAHYQY8ceFc8992yL9YMPHhFPPPHb/IEAANjhVFf3iqqqts+/OiELAECn9k6MraioiM985p/j//7fp+Izn/nnqKioiOeeezbGjj2q3CMCANCBOCELAECntWrVqthrrz2ioqIi5s9fHD169Gh6tnbt2hg0aEAUi8WYM2dB7LzzzmWcFACA7Z0TsgAA0IYvfemLERFx0kmfbRZjIyJ69OgRJ5zwmWb7AABgawmyAAB0WvPmzY2IiHPPPb/V5+ec8+Vm+wAAYGsJsgAAdFqDBw+JiIjvf//WVp//x3/c1mwfAABsLXfIAgDQablDFgCAUnGHLAAAtGHnnXeOgw8eEcViMQYNGhDnnPOFeP755+Kcc77QFGMPPniEGAsAQMk4IQsAQKc3duxR8dxzz7ZYP/jgEfHEE7/NHwgAgB3O5p6QFWQBACDevr7gS1/6YsybNzcGDx4S3/veXU7GAgCw2QRZQRYAAAAASOIOWQAAAACA7YwgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAEASQRYAAAAAIIkgCwAAAACQRJAFAAAAAEgiyAIAAAAAJBFkAQAAAACSCLIAAAAAAEkEWQAAAACAJIIsAAAAAECSrQqyd9xxR4wfP77Z2lVXXRX77bdfs58xY8a86+f88pe/jGOPPTYOPPDAOP744+Ppp5/emrEAAKDd5s6dGzU1/aN///dETU3/mDt3brlHAgCgA+qypW986KGH4uabb46RI0c2W581a1acffbZceqppzatVVVVbfJz6urq4tJLL43LLrssPvShD8V//dd/xZlnnhk/+9nPYujQoVs6HgAAbLaBA/tGY2Nj0+u1a9fGqFEHRWVlZSxatLx8gwEA0OG0+4Ts4sWL4+yzz44bb7wxBg8e3OxZsViMl19+OQ444IDYbbfdmn6qq6s3+Xl33XVXHH300XHaaafF0KFD4/LLL4/3ve99cd9997X7HwMAAO21cYzt3btPXHfdv0fv3n0iIqKxsTEGDuxbxukAAOho2h1k//rXv0bXrl1j0qRJcdBBBzV7Vl9fH2vWrIm99tprsz6rsbExnn322TjssMOarY8aNSqmT5/e3tEAAKBd5s6d2xRj//KXl+OVV/4WX/jCWfHKK3+Lv/zl5Yh4+3dW1xcAAFAq7b6yYMyYMZu8E3b27NkREfHAAw/E73//+6isrIwjjjgiLrrooujdu3eL/StWrIg1a9bEwIEDm633798/Fi1a1N7RAACgXY48clREvH0ytn///s2e9e/fP3r37h0rV66MI48cFfX1S8oxIgAAHcxWfanXP5o9e3ZUVlZG//794/bbb4+vfOUr8eSTT8a5557b7E6ud6xduzYiIrp169ZsvXv37rFu3bpSjgYAAC2sXfv275xXXPGvrT6/+OLLm+0DAICttcVf6tWac845Jz73uc9Fv379IiJi3333jd122y3+6Z/+Kf785z+3uOKge/fuERGxfv36Zuvr1q2LnXbaqZSjAQBACz16dI+1a9fG9dd/I77whbNaPL/ppm817QMAgFIo6QnZysrKphj7jn322SciotUrCPr27Rs9e/aMJUua//nXkiVLYsCAAaUcDQAAWvjd76ZFRMTKlSta/Z105cqVzfYBAMDWKmmQveyyy+L0009vtvbnP/85IiL23nvvFvsrKipixIgR8cwzzzRbnzZtWowcObKUowEAQAtDhgyJysq3fyU+4IC9Y+jQ98b3vndrDB363jjggLd/f62srIwhQ4aUc0wAADqQkgbZY445Jp5++um47bbbor6+Pn73u9/FV7/61TjuuONi6NChERGxcuXKaGhoaHrPGWecEb/4xS/innvuiVdeeSVuuOGGmDlzZkyYMKGUowEAQKsWLVreFGVXrlwZV199VdPJ2MrKyli0aHkZpwMAoKMpaZD92Mc+FjfffHP8+te/jk9+8pNx5ZVXxtixY+O6665r2nPttdfGSSed1PT6wx/+cFx33XXxyCOPxLhx46Kuri5uv/32poALAADb2qJFy2PatP+JHj16RERF9OjRI6ZN+x8xFgCAkqsoFovFcg9RaoVCYzQ0rC73GAAAAABAJ1Fd3Suqqto+/1rSE7IAAAAAAGyaIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAABFRX18fQ4bsHgMG9I0hQ3aP+vr6co8EAEAHVFEsFovlHqLUCoXGaGhYXe4xAADYQeyxR3Vs2LChxXqXLl1iwYKGMkwEAMCOprq6V1RVtX3+1QlZAAA6tY1jbL9+1fHtb98a/fpVR0TEhg0bYo89qss5HgAAHUyXcg8AAADlUl9f3xRjX3hhTuy6664RETF+/Onx+uuvx/Dhe8WGDRuivr4+ampqyjkqAAAdhBOyAAB0WkceOSoi3j4Z+06Mfceuu+4a/fr1a7YPAAC2liALAECntWbNmxERcdVVta0+v/TSK5vtAwCAreVLvQAA6LSGDNk9Vq9eHf36VcesWfNaPN9vv0GxbNmy6NWrV8yduzB/QAAAdhi+1AsAANrwu99Ni4iIZcsa4vXXX2/27PXXX49ly5Y12wcAAFtLkAUAoNOqqamJLl3e/p7b4cP3iv32GxQ/+MGdsd9+g2L48L0iIqJLly6+0AsAgJJxZQEAAJ3eHntUx4YNG1qsd+nSJRYsaCjDRAAA7GhcWQAAAJtpwYKGmDHjL9GrV6+oqKiMXr16xYwZfxFjAQAoOSdkAQAAAAC2khOyAAAAAADbGUEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAARMRjj02J/v37NP089tiUco8EAEAHVFEsFovlHqLUCoXGaGhYXe4xAADYQfTv32eTz5YsWZE4CQAAO6rq6l5RVdX2+VcnZAEA6NT+McYeeeRH3/U5AABsDUEWAIBOa+NrCR577LexZMmK+PGPfx5LlqyIxx77bav7AABga7iyAACATmvj06+tXU3Q1nMAAHiHKwsAAGAz/eM1Be8YNerw5EkAAOjonJAFAKDTckIWAIBScUIWAADacP/9/9n0/88++2yzZxu/3ngfAABsDSdkAQDo1DY+BRvx9jUF06Y91WzN6VgAANqyuSdkBVkAADq9f4yyGxNjAQDYHK4sAACAzbRkyYoW1xLcf/9/irEAAJScE7IAAAAAAFvJCVkAAAAAgO2MIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJBFkAAAAAgCSCLAAAAABAEkEWAAAAACCJIAsAAAAAkESQBQAAAABIIsgCAAAAACQRZAEAAAAAkgiyAAAAAABJtirI3nHHHTF+/Phma7/5zW/ixBNPjA984AMxZsyY+Na3vhVr167d5GcUCoU48MADY7/99mv2893vfndrRgMAAAAA2O502dI3PvTQQ3HzzTfHyJEjm9ZmzJgRX/7yl+P888+PT3ziEzF//vz42te+FsuXL4/rr7++1c+ZN29erFu3Ln7+85/HLrvs0rTes2fPLR0NAAAAAGC71O4gu3jx4vj6178e06ZNi8GDBzd79p//+Z8xatSoOPvssyMiYvDgwXHRRRfFVVddFVdffXV069atxefNmjUrdt5559h///237F8AAAAAALCDaHeQ/etf/xpdu3aNSZMmxfe+97147bXXmp59/vOfj8rK5rcgVFZWxltvvRWrVq2K6urqFp83a9asGDp06BaMDgAAAACwY2l3kB0zZkyMGTOm1WfDhw9v9vqtt96Ke++9Nw444IBWY2xExOzZs2PDhg0xceLEePHFF2PAgAExYcKE+PSnP93e0QAAAAAAtmtbfIdsWzZs2BCXXXZZvPTSS/HQQw9tct9LL70UjY2Ncf7558fAgQPjd7/7XVxxxRXx1ltvxUknnbStxgMAAAAASLdNguyqVaviwgsvjGeeeSZuu+22OPDAAze5d/LkyVEoFKJXr14REbH//vvHggUL4u677xZkAQAAAIAOpbLtLe2zZMmSOOWUU+K5556Lu+++O4488sh33d+jR4+mGPuOfffdNxYtWlTq0QAAAAAAyqqkQfbvf/97TJgwIRoaGuKhhx6KQw455F33r1ixIg499NB49NFHm63/+c9/jn322aeUowEAAAAAlF1Jryy4/vrr49VXX40f/OAHUV1dHUuXLm16Vl1dHVVVVbF8+fKIiOjbt2/06dMnRo8eHd/5zndil112iUGDBsUTTzwRkyZNijvuuKOUowEAAAAAlF1FsVgsbumbv/KVr8Rrr70WDzzw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a+7o6FBaWppCQ0MVFBSkzMxM9fT03NV9u3TpklatWqWgoCAFBgZq1apVqq6u7nMP6urq5Ofn1+/rk08+kST98ccf2rp1q6ZPn66goCCtWLFC33333V31AwAAgMFHIAsAAID79uGHHyoyMlKlpaVasmSJsrKyVFFRIUk6fvy43njjDc2cOVPFxcXavn27SktLlZCQMODzs7OzVV1drby8PJWWlurZZ59VfHy8eZ7rO++8o6KiIm3dulVHjx7VihUrlJycrAMHDlics3//fnONj4+P1q9fr9bWVklSRkaGcnNztXbtWh09elTR0dFKTU1VQUFBvz29/fbbKi0tVUZGhg4ePKiGhgadOXPmru5bQkKCxowZo48//liHDx/WsGHDtHbt2j51np6eKi8vN79OnDihJ598UhMnTtTs2bNlMpkUGxury5cva8+ePTp06JACAgK0ZMkSnTt37q56AgAAwOBiZAEAAADu29KlSxUZGSlJWr16tfbt26fvv/9eoaGhysvL0+zZs7V69WpJ0vjx42UymbRmzRr99NNPeuyxx/7x/NraWo0YMULjxo3TyJEjFRcXp+DgYLm5uenmzZsqKChQVlaWZs6cKUny9vbWL7/8or179yo6Otp8zubNm/XUU09JktasWaPPPvtMP//8syZMmKCioiJt3LhRCxYskCT5+Piorq5OeXl5euWVVyz6aWlpUXFxsbZt26bnnntOkpSWlqZTp07d1X2rra3V008/LS8vLzk6OiotLU0XLlxQT0+Phg37/7MT9vb2FuMLUlNTdenSJR06dEguLi6qqKjQ119/rVOnTmnUqFGS/gx7z549q/379ysjI+Ou+gIAAMDgIZAFAADAffPx8bH4euTIkWpvb5ck/fjjj5o/f77FfkhIiHlvIIFsbGysXn/9dYWGhuqJJ57QjBkztGDBArm6uurbb79Ve3u7EhMTLULMrq4udXR0WIxGGD9+vEWPknTr1i1duHBBnZ2dCgoK6tPnBx98oKamJov1ixcvqrOzU/7+/uY1g8GgyZMn/+O1/FV8fLzS0tJUWFiokJAQhYWFKSIiwuI6/u7AgQP66KOPtH//fnl5eUmSfvjhB5lMJs2aNcuitqOjw/x7AAAAwL8DgSwAAADum729fZ81k8lk8fGvemetOjj0/+doV1eXxdeBgYEyGo364osvVFFRoSNHjmj37t16//33NXz4cEnSu+++qwkTJvQ5y8nJyfx5f0GnyWTqt8c79WlnZ9fvtd3uem4nOjpac+fOldFoVEVFhXJycrR7924dOXJEDz/8cJ/6EydOKC0tTRkZGQoICLDo08XFpc8cXsny+gEAADD0mCELAACAQeXn56ezZ89arPXOWvX19ZUkOTo6qqWlxbxfU1NjUZ+Tk6OqqiqFh4frrbfe0rFjxzRu3DgdO3ZMEyZMkIODg+rr6/Xoo4+aX0ajUXv37r3j06a9fH195ejoqKqqqj59enh4yM3NzWJ9/PjxMhgMFtfV1dWl8+fPD+CO/KmpqUk7duxQZ2enFi1apMzMTJWUlKixsVGnT5/uU19dXa34+Hi9+uqr5rEKvSZOnKiWlhZ1dnZa3IP8/Hx9/vnnA+4JAAAAg48nZAEAADCoYmJiFBcXp9zcXM2bN0+XLl1SSkqKZs2aZQ5kAwICdPjwYQUHB8tkMik9Pd3iyc7Lly+rpKREKSkp8vb21jfffKP6+noFBgbK1dVVixcvVnZ2tlxcXDRt2jRVVlYqMzNTr7322oB6dHFx0csvv6ycnByNGjVK/v7+Ki8vV2FhoRISEsxPxPYaMWKEli1bppycHHl4eMjX11f79u3TlStXBnxf3NzcVFZWptraWiUmJpqfcHV0dNSUKVMsahsbG80jG5YvX67GxkbznrOzs8LCwjRp0iTFx8dry5Yt8vT0VGFhoYqLi7V3794B9wQAAIDBRyALAACAQTVnzhxlZWVp9+7dys3Nlbu7uyIiIrRu3TpzTXJyspKTkxUVFaXRo0crLi5ODQ0N5v1t27Zp586devPNN9Xc3CwvLy9t2LBBCxculCRt2rRJDz74oLKzs3X16lV5enpq3bp1iomJGXCfvWfs2rVLv/32m3x8fJSUlKSoqKh+6xMTE2UwGLRjxw61trZq3rx5ev755wf88xwcHJSfn6+dO3dq5cqVamtr06RJk5SXlydvb2+L2pMnT6q+vl719fU6fvy4xd6LL76ojIwM7du3T5mZmVq/fr3a2trk6+ur9957T6GhoQPuCQAAAIPPznS7gVkAAAAAAAAAAKtihiwAAAAAAAAA2AgjCwAAAAAry8/PV25u7h1rNm/erJdeeslGHQEAAODfgpEFAAAAgJVdv35dzc3Nd6x56KGH5OLiYpuGAAAA8K9BIAsAAAAAAAAANsIMWQAAAAAAAACwEQJZAAAAAAAAALARAlkAAAAAAAAAsBECWQAAAAAAAACwEQJZAAAAAAAAALARAlkAAAAAAAAAsBECWQAAAAAAAACwEQJZAAAAAAAAALCR/wFmNb8+GFOMoQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1700x1500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABWAAAASyCAYAAAA4dkJNAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAABOoklEQVR4nOzdfZTWdZ34/xcMIg4KBQGjizCoKWYK3oC4qRkaq0mtaPjLRXTxDpSyMo83i6RodidfM80sJW9YNZUDR/FmkdhsW0skNLSjkqEMijKOioKIiM7M7w+Xaa5hVGiuFxdwPR7nzIn5vD8zvugvz/O8fV3tGhsbGwMAAAAAgKJrX+oBAAAAAAC2VgIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJB1KPUAxNDY2RkNDY6nHAAAAAADKQPv27aJdu3Yb9O5WEWAbGhpj+fK3Sz0GAAAAAFAGunXrHBUVGxZgrSAAAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJG0KsL/85S9j9OjRBc+eeeaZOPHEE2PgwIExdOjQmDp1asF5Q0NDXH311XHIIYfEwIED4/TTT48XX3yxLWMAAMBGWbFiRQwfPiwGDtwzhg8fFitWrCj1SAAAbKX+4QB72223xVVXXVXw7I033ogxY8ZEnz59Yvr06TF+/PiYPHlyTJ8+vemdn//853H77bfHZZddFnfccUc0NDTEaaedFmvXrv2H/xIAALChBg8eEJ/+9M4xb97cePnll2LevLnx6U/vHIMHDyj1aAAAbIU2OsC+8sorMW7cuJg8eXJUV1cXnN11112xzTbbxKWXXhq77rprHHfccfHv//7vcf3110dExNq1a+PGG2+Ms88+Ow477LDo379//OQnP4na2tqYPXt2Uf5CAADwYQYPHhA1NYsjImLo0CPigQfmxNChR0RERE3NYhEWAICi2+gA+9RTT8U222wTM2fOjAEDCv8Fdf78+TF48ODo0KFD07MhQ4ZETU1NvPbaa7Fw4cJ4++2346CDDmo679KlS3zmM5+JP/3pT234awAAwEdbsWJFU3ytqamNO+6YEQccMDjuuGNG1NTU/t/zxdYRAABQVBsdYIcOHRrXXHNN7Lzzzuud1dbWRlVVVcGznj17RkTEsmXLorb2g3+x3XHHHdd7Z90ZAABkGDVqZER8cPO1srKy4KyysjIOO2xowXsAAFAMbfoQrpbWrFkTHTt2LHi27bbbRkTEu+++G++8805ERKvvvPvuu8UcBQAACixd+sEHv5577gWtnp9zznkF7wEAQDEUNcB26tRpvQ/TWhdWKysro1OnThERrb6z3XbbFXMUAAAo0Lv3B/8F1+TJP2z1/Morf1zwHgAAFENRA2xVVVXU1dUVPFv3fa9evZpWD7T2Tq9evYo5CgAAFLjttmkREfHb386J1atXF5ytXr06fve73xa8BwAAxVDUADto0KB47LHHor6+vunZ3Llzo1+/ftG9e/fo379/bL/99vHoo482na9cuTKefvrpGDRoUDFHAQCAAl27do3q6n4REVFdXRXHH39MzJ37xzj++GOiurrq/573i65du5ZyTAAAtjJFDbDHHXdcrFq1KiZMmBCLFi2KGTNmxM033xxjx46NiA92v5544okxefLk+O///u9YuHBhfPvb346qqqoYNmxYMUcBAID1zJv3RFOE/d3vfhtf+cqRTTdfq6v7xbx5T5RyPAAAtkIdivnLunfvHlOmTInLL788RowYET169IjzzjsvRowY0fTO2WefHe+//35cdNFFsWbNmhg0aFD86le/im222aaYowAAQKvmzXsiVqxYEaNGjYylS1+M3r13jttum+bmKwAAKdo1NjY2lnqItqqvb4jly98u9RgAAAAAQBno1q1zVFRs2HKBoq4gAAAAAADg74q6ggAAALYEa9eujZtuuiFqahZHdXW/GDPm9OjYsWOpxwIAYCtkBQEAAGVl0qSJ8Ytf/Czq6+ubnlVUVMS4cV+Piy++rISTAQCwpbCCAAAAWjFp0sS49tqfRrdu3ePKK6+Ov/zlb3HllVdHt27d49prfxqTJk0s9YgAAGxl3IAFAKAsrF27Nvr27RXdunWPJ55YGB06/H0b1/vvvx8DBvSP5cuXx5IltdYRAADwkdyABQCAFm666Yaor6+PCy+8qCC+RkR06NAhzj9/QtTXvx833XRDiSYEAGBrJMACAFAWamoWR0TEF794VKvnw4YdWfAeAAAUgwALAEBZqK7uFxERv/nNf7V6Pnv2rIL3AACgGOyABQCgLNgBCwBAsdgBCwAALXTs2DHGjft6vPpqXQwY0D+mTr0pamuXxdSpN8WAAf3j1VfrYty48eIrAABF5QYsAABlZdKkifGLX/ws6uvrm55VVHSIcePGx8UXX1bCyQAA2FJszA1YARYAgLKzdu3auOmmG6KmZnFUV/eLMWNOd/MVAIANJsACAAAAACSxAxYAAD7C8uXL49BDD4zdd+8bhx56YCxfvrzUIwEAsJVyAxYAgLKy1167xauv1q33vEePnvHUU4tKMBEAAFsaN2ABAKAVzePr/vsPiunT74399x8UERGvvloXe+21WynHAwBgK+QGLAAAZWH58uXRv391REQ8//zLsf322zedrVq1KnbZZaeIiFi4sCa6detWihEBANhCuAELAAAtHHPMURHxwc3X5vE1ImL77bePfffdv+A9AAAoBgEWAICyUFtbGxER//Ef3231/PzzJxS8BwAAxSDAAgBQFqqqqiIi4vvfv7TV8x/96PKC9wAAoBjsgAUAoCzYAQsAQLHYAQsAAC1069YtevToGRERu+yyU/zLv3whfvvbOfEv//KFpvjao0dP8RUAgKJyAxYAgLKy1167xauv1q33vEePnvHUU4tKMBEAAFsaN2ABAOBDPPXUoli4sCb6998zPvGJT0b//nvGwoU14isAACncgAUAAAAA2AhuwAIAAAAAbAY6lHoAAADY1Orr62Pu3D/GK6/URq9eVTFkyD9HRUVFqccCAGArJMACAFBW7rtvZlxyyYR44YUlTc/69Okbl1xyeQwf/pUSTgYAwNbICgIAAMrGfffNjFNPHR177vmZeOCBOfH88y/H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AQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJAIsAAAAAEASARYAAAAAIIkACwAAAACQRIAFAAAAAEgiwAIAAAAAJBFgAQAAAACSCLAAAAAAAEkEWAAAAACAJCkB9u67744vfelLsffee8fRRx8d//Vf/9V0tnTp0hg7dmzst99+cfDBB8dVV10V9fX1GWMAAAAAAJRU0QPsPffcExMmTIhRo0bF/fffH8OHD49zzjkn/vznP8d7770Xp556akRE3HHHHXHJJZfEr3/967j22muLPQYAAAAAQMl1KOYva2xsjJ/+9Kdx0kknxahRoyIi4swzz4z58+fHvHnz4qWXXoqXX3457rrrrujatWvsvvvu8frrr8ePf/zjGDduXHTs2LGY4wAAAAAAlFRRb8AuXrw4Xnrppfjyl79c8PxXv/pVjB07NubPnx977bVXdO3atelsyJAhsWrVqnjmmWeKOQoAAAAAQMkVPcBGRKxevTpOPfXUOOigg2LkyJHx29/+NiIiamtro6qqquBnevbsGRERy5YtK+YoAAAAAAAlV9QAu2rVqoiIOP/882P48OFx4403xuc+97k466yz4pFHHok1a9ast2Zg2223jYiId999t5ijAAAAAACUXFF3wG6zzTYREXHqqafGiBEjIiJizz33jKeffjpuuumm6NSpU6xdu7bgZ9aF18rKymKOAgAAAABQckW9AdurV6+IiNh9990Lnu+2226xdOnSqKqqirq6uoKzdd+v+1kAAAAAgK1FUQPsXnvtFZ07d44nnnii4Pmzzz4bffr0iUGDBsXTTz/dtKogImLu3LnRuXPn6N+/fzFHAQAAAAAouaIG2E6dOsVpp50W1157bdx3333xwgsvxHXXXRd/+MMfYsyYMXHEEUdEjx494lvf+lYsXLgw5syZE1deeWWccsop6+2GBQAAAADY0hV1B2xExFlnnRXbbbdd/OQnP4lXXnkldt11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",
      "text/plain": [
       "<Figure size 1700x1500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "def plot_boxplot(df, ft):\n",
    "  df.boxplot(column=[ft])\n",
    "  plt.grid(False)\n",
    "  plt.show()\n",
    "\n",
    "for feat in num_features:\n",
    "  plot_boxplot(data, feat)\n",
    "  print(\"\\n\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a function that returns a list containing all the index of outliers based on interquatile range\n",
    "def outliers(df, ft):\n",
    "  Q1 = df[ft].quantile(0.25)\n",
    "  Q3 = df[ft].quantile(0.75)\n",
    "  IQR = Q3 - Q1\n",
    "\n",
    "  lower_bound = Q1 - 1.5 * IQR\n",
    "  upper_bound = Q3 + 1.5 * IQR\n",
    "\n",
    "  index_list = df.index[ (df[ft] < lower_bound) | (df[ft] > upper_bound) ]\n",
    "\n",
    "  return index_list\n",
    "\n",
    "index_lists=[]\n",
    "for feature in num_features:\n",
    "    index_list= outliers(data,feature)\n",
    "    if len(index_list) > 0:\n",
    "        index_lists.extend(index_list)\n",
    "index_lists = sorted(set(index_lists))\n",
    "data = data.drop(index_lists)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABX4AAASyCAYAAAABEfMWAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAABeRklEQVR4nOzdf5TXdd3n/8cgMkKBMigCUR4dlXbzRw5QUqlYom6eQrxW90q0Uiiu2guvQEWIRJHFwOE7apdLpmSKhLm2sSe0lQsOlxu6EIKdi+u6UAykWTKRUSAg+aHMfP/oMMcJrPm0+/3CvPZ2O8dzfP9+fj7nzDl1P29fn6qWlpaWAAAAAABQjE6HewAAAAAAAP7PEn4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIXpfLgH6MhaWlrS3NxyuMcAAAAAAP4v0KlTVaqqqtp1rvD7v6G5uSVbt/7+cI8BAAAAAPxfoKbmfTnqqPaFX0s9AAAAAAAURvgFAAAAACiM8AsAAAAAUBjhFwAAAACgMMIvAAAAAEBhhF8AAAAAgMIIvwAAAAAAhRF+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8AgAAAAAURvgFAAAAACiM8AsAAAAAUBjhFwAAAACgMMIvAAAAAEBhhF8AAAAAgMIIvwAAAAAAhRF+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYSoOv9u3b8+UKVNy/vnnp66uLl/4wheyatWq1uPLly/PFVdckbPPPjuXXnppnnrqqfe815QpUzJx4sSD9v/P//k/81d/9Vf56Ec/mosuuijf//73/+xcP/zhD/OZz3wmZ511Vq6++uqsXbu2zfHf/OY3GTNmTOrq6vKpT30q99xzT/bv31/BJwcAAAAA6BgqDr/jx4/PL3/5yzQ0NOS//tf/mn/zb/5NRo0alVdeeSUbNmzImDFjct555+UnP/lJrrzyykyYMCHLly9vc4/m5uY0NDTk8ccfP+j+r7zySsaMGZMLL7wwCxcuzPjx4/Od73wnP/zhD99zpgULFuSuu+7K3/3d3+UnP/lJ+vfvn+uuuy5bt25Nkrz99tsZNWpUkuRHP/pRbr/99jz22GP5z//5P1f68QEAAAAAjnidKzm5sbExzz33XObPn5+BAwcmSW699dYsW7YsCxcuzJtvvpkBAwZk3LhxSZLa2tqsXbs2c+bMyZAhQ5IkGzZsyOTJk9PY2Jh+/fod9Iyf//zn6datW/72b/82SfLBD34wP/vZz7Js2bKMHDnykHPdf//9ueaaa/L5z38+SXLnnXfmoosuyhNPPJExY8Zk0aJF+e1vf5v/8l/+S4499ticfvrpefPNN3PXXXflb/7mb9KlS5dKvgYAAAAAgCNaRW/89uzZMw888EDOPPPM1n1VVVWpqqrKjh07smrVqtbAe8C5556b1atXp6WlJUmyYsWK1NbW5sknn0z//v0PekavXr2yffv2PPnkk2lpacm6deuyevXqnH322Yec6c0338yvf/3rNs/t3LlzBg0alOeffz5JsmrVqnzkIx/Jscce22auXbt25cUXX6zkKwAAAAAAOOJV9MZvjx49csEFF7TZt2jRojQ2Nuab3/xmFixYkD59+rQ53rt37+zevTvbtm1LTU3Ne761e8C/+3f/Lr/4xS9y8803Z8KECdm/f38+97nP5W/+5m8Oef7mzZuTJH379j3ouS+99FLrOYeaK0lee+2194zKAAClef3117Jnz57DPcb/ls2bX8vvf//7wz0GSd73vvelT5++f/7EI9gxxxyTE0/s2J8BAOBQKgq/f+yFF17IpEmTcvHFF2fo0KHZs2fPQcsmHNjet29fu+755ptv5tVXX80NN9yQCy64IGvXrs3MmTPz93//97nhhhsOOn/37t1tnnNAdXV19u7dmyTZs2dPevTocdDxJK3nAACU7vXXX8ukSTce7jH+t7zzzjtZs2bN4R6DdznrrLPSufP/1v+tOOy+/e3/R/wFAIrzF/8vtCVLluSmm25KXV1dZs2aleQPMfWPA++B7a5du7brvpMnT07fvn3zta99LUnyb//tv01LS0tuv/32XHPNNampqWlz/jHHHNPmOQfs3bu39ZnHHHPMIY8nSbdu3do1FwBAR3fgTd+vfOXr6dfvA4d5mr+cN36PHB39jd/f/vbVPPjg7A7/FjwAwKH8ReF33rx5mT59ei699NLMnDmz9W3bvn37ZsuWLW3O3bJlS7p165bu3bu3696rV6/OhAkT2uz76Ec/mnfeeSe/+c1vDgq/B5Z42LJlS2pra9s898QTT0yS9OnTJy+//PJBcyVpPQcA4P8W/fp9ICeddPLhHuMv1pFnBwCA/79U9ONuSTJ//vxMmzYtI0eOTENDQ5slFgYNGpSVK1e2OX/FihWpq6tLp07te9SJJ56YdevWtdm3bt26VFVV5aSTTjro/F69euXkk0/OL37xi9Z977zzTlatWpXBgwcnSQYPHpy1a9dm165dbeZ63/velw9/+MPtmgsAAAAAoKOoKPxu3Lgxd955Z4YNG5YxY8bkjTfeSFNTU5qamrJz585ce+21WbNmTWbNmpUNGzbkoYceytNPP53Ro0e3+xnXXXddnnjiicydOzebNm3KkiVLMmPGjFx99dU59thjkyTbt2/P9u3bW6+5/vrr84Mf/CALFizI+vXr881vfjN79uzJv//3/z5JctFFF+WEE07IN77xjbz00ktZsmRJGhoacv311x+0NjAAAAAAQEdX0VIPixYtyttvv53Fixdn8eLFbY6NGDEiM2bMyOzZs1NfX59HHnkk/fv3T319fYYMGdLuZ/yH//AfUl1dnR/84AdpaGjIiSeemKuvvjpf+cpXWs8ZO3ZskuTRRx9Nklx11VXZuXNn7rnnnmzfvj1nnHFGfvCDH7QuC1FdXZ05c+Zk6tSpueqqq3Lsscfm6quvzte//vVKPj4AAAAAQIdQ1dLS0nK4h+io9u9vztatflgEAOgYGhs3ZurUybnttunWyYX4mwAAOp6amvflqKPat4hDxWv8AgAAAABwZBN+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8AgAAAAAURvgFAAAAACiM8AsAAAAAUBjhFwAAAACgMMIvAAAAAEBhhF8AAAAAgMIIvwAAAAAAhRF+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8AgAAAAAURvgFAAAAACiM8AsAAAAAUBjhFwAAAACgMMIvAAAAAEBhhF8AAAAAgMIIvwAAAAAAhRF+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8AgAAAAAURvgFAAAAACiM8AsAAAAAUBjhFwAAAACgMMIvAAAAAEBhhF8AAAAAgMIIvwAAAAAAhRF+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8AgAAAAAURvgFAAAAACiM8AsAAAAAUBjhFwAAAACgMMIvAAAAAEBhhF8AAAAAgMIIvwAAAAAAhRF+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8AgAAAAAURvgFAAAAACiM8AsAAAAAUBjhFwAAAACgMMIvAAAAAEBhOldy8vbt29PQ0JBnnnkmu3btyoABA3LjjTdm0KBBSZLly5envr4+GzZsSN++fTN27Nhcdtllh7zXlClTsm/fvsyYMaN136c//em8+uqrhzx/3rx5GTx48EH7BwwY8J7z/uM//mP69euX1atX5+qrrz7o+Ny5c/Pxj3/8T35mAAAAAICOpqLwO378+DQ1NaWhoSG9evXKo48+mlGjRmXBggVpaWnJmDFjct1116W+vj7PPPNMJkyYkJqamgwZMqT1Hs3Nzbnnnnvy+OOPZ8SIEW3u/+Mf/zj79+9v3d63b1+uv/769OnTJ+ecc84hZ3r22WfbbP/ud7/LNddckwsuuCD9+vVLkqxbty4f+tCHMn/+/DbnHnvssZV8fAAAAACADqHd4bexsTHPPfdc5s+fn4EDByZJbr311ixbtiwLFy7Mm2++mQEDBmTcuHFJktra2qxduzZz5sxpDb8bNmzI5MmT09jY2Bpl362mpqbN9syZM7Njx4489thj6dz50KOecMIJbbanTZuWnj17Ztq0aa37Xn755Zx66qkHnQsAAAAAUKJ2r/Hbs2fPPPDAAznzzDNb91VVVaWqqio7duzIqlWr2rzZmyTnnntuVq9enZaWliTJihUrUltbmyeffDL9+/f/k89bv3595s6dm4kTJx4UhN/Ls88+m3/4h3/ItGnT0qVLl9b969atS21tbXs/KgAAAABAh9buN3579OiRCy64oM2+RYsWpbGxMd/85jezYMGC9OnTp83x3r17Z/fu3dm2bVtqamoycuTIdg/2ne98J6effnqGDx/e7msaGhrymc98pnXN4QN+9atfpWfPnrniiivy+uuv5/TTT8+4ceNy1llntfveAAAAAAAdRbvf+P1jL7zwQiZNmpSLL744Q4cOzZ49e9q8ZZukdXvfvn0V3XvTpk1ZvHhxvva1r7X7mueffz7/+q//mq9//ett9r/22mvZuXNn3nrrrXzrW9/K7Nmzc/zxx+eaa67J+vXrK5oLAAAAAKAjqOjH3Q5YsmRJbrrpptTV1WXWrFlJkurq6oMC74Htrl27VnT/n/70p+nVq1cuuuiidl+zYMGCnHXWWfnIRz7SZn/fvn3z/PPPp2vXrjn66KOTJGeeeWbWrl2bRx99NFOnTq1oNgAAAACAI13Fb/zOmzcvY8eOzYUXXpj7778/1dXVSf4QWLds2dLm3C1btqRbt27p3r17Rc9YsmRJLrvssnTq1L7xmpubs3Tp0nzuc5875PEePXq0Rt8k6dSpU2pra/P6669XNBcAAAAAQEdQUfidP39+pk2blpEjR6ahoaHN0g6DBg3KypUr25y/YsWK1NXVtTvgJsmuXbvy4osv5hOf+ES7r1m/fn22bdt2yGt+/vOf55xzzsmmTZta973zzjt56aWXcuqpp7b7GQAAAAAAHUW7i+zGjRtz5513ZtiwYRkzZkzeeOONNDU1pampKTt37sy1116bNWvWZNasWdmwYUMeeuihPP300xk9enRFA7300ktpaWnJhz/84UMe37lzZ7Zu3dpm39q1a3P00UfnlFNOOej8urq69OzZM7fcckv+5V/+JevWrcstt9yS7du358tf/nJFswEAAAAAdATtXuN30aJFefvtt7N48eIsXry4zbERI0ZkxowZmT17durr6/PII4+kf//+qa+vz5AhQyoa6MByEccdd9whj0+fPj0rV67M0qVLW/c1NTXl2GOPPeSbxe9///vz8MMPZ9asWRk1alT27t2bgQMHZt68eTn++OMrmg0AAAAAoCOoamlpaTncQ3RU+/c3Z+vW3x/uMQAA2qWxcWOmTp2c226bnpNOOvlwjwOHnb8JAKCjqal5X446qn2LOFT8424AAAAAABzZhF8AAAAAgMIIvwAAAAAAhRF+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8AgAAAAAURvgFAAAAACiM8AsAAAAAUBjhFwAAAACgMMIvAAAAAEBhhF8AAAAAgMIIvwAAAAAAhRF+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8AgAAAAAURvgFAAAAACiM8AsAAAAAUBjhFwAAAACgMMIvAAAAAEBhhF8AAAAAgMIIvwAAAAAAhRF+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8AgAAAAAURvgFAAAAACiM8AsAAAAAUBjhFwAAAACgMMIvAAAAAEBhhF8AAAAAgMIIvwAAAAAAhRF+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8AgAAAAAURvgFAAAAACiM8AsAAAAAUBjhFwAAAACgMMIvAAAAAEBhhF8AAAAAgMIIvwAAAAAAhRF+AQAAAAAKI/wCAAAAABRG+AUAAAAAKIzwCwAAAABQGOEXAAAAAKAwwi8AAAAAQGGEXwAAAACAwgi/AAAAAACFEX4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8AgAAAAAURvgFAAAAACiM8AsAAAAAUJjOlV6wffv2NDQ05JlnnsmuXbsyYMCA3HjjjRk0aFCSZPny5amvr8+GDRvSt2/fjB07Npdddtkh7zVlypTs27cvM2bMaN336U9/Oq+++uohz583b14GDx58yGMXX3xxGhsb2+wbMWJE6723bduW//Sf/lN+/vOfp6qqKpdddlkmTJiQrl27VvoVAAAAAAAc0SoOv+PHj09TU1MaGhrSq1evPProoxk1alQWLFiQlpaWjBkzJtddd13q6+vzzDPPZMKECampqcmQIUNa79Hc3Jx77rknjz/+eEaMGNHm/j/+8Y+zf//+1u19+/bl+uuvT58+fXLOOecccqa33normzZtyve+97185CMfad1/zDHHtP77DTfckN27d+fhhx/Ojh07Mnny5Lz11luZOXNmpV8BAAAAAMARraLw29jYmOeeey7z58/PwIEDkyS33nprli1bloULF+bNN9/MgAEDMm7cuCRJbW1t1q5dmzlz5rSG3w0bNmTy5MlpbGxMv379DnpGTU1Nm+2ZM2dmx44deeyxx9K586HHXb9+fZqbm3POOefk2GOPPej4L3/5y6xcuTI/+9nPUltbmyS54447Mnr06IwfPz4nnnhiJV8DAAAAAMARraI1fnv27JkHHnggZ555Zuu+qqqqVFVVZceOHVm1alWbN3uT5Nxzz83q1avT0tKSJFmxYkVqa2vz5JNPpn///n/yeevXr8/cuXMzceLEg4Lwu61bty7HH3/8IaNvkqxatSonnHBCa/RNko997GOpqqrK6tWr/+znBgAAAADoSCp647dHjx654IIL2uxbtGhRGhsb881vfjMLFixInz592hzv3bt3du/enW3btqWmpiYjR45s9/O+853v5PTTT8/w4cP/5Hnr1q1Lt27dcsMNN+SFF15Iz54981d/9Vf54he/mE6dOuX1119P375921zTpUuXHHfccXnttdfaPQ8AAAAAQEdQ8Rq/7/bCCy9k0qRJufjiizN06NDs2bMnXbp0aXPOge19+/ZVdO9NmzZl8eLFuffee//sub/61a+yY8eOXHLJJfmP//E/ZvXq1amvr8/vfve7/N3f/V1279590FxJUl1dnb1791Y0FwAAAADAke4vDr9LlizJTTfdlLq6usyaNSvJH0LqHwfeA9tdu3at6P4//elP06tXr1x00UV/9twHH3wwe/fuTffu3ZMkAwYMyK5du/Ld7343Y8eOzTHHHHPI8Lx3795069atorkAAAAAAI50Fa3xe8C8efMyduzYXHjhhbn//vtTXV2dJOnbt2+2bNnS5twtW7akW7durVG2vZYsWZLLLrssnTr9+RG7dOly0P1PP/30vPXWW/nd736XPn36HDTXvn37sn379vTu3buiuQAAAAAAjnQVh9/58+dn2rRpGTlyZBoaGtosoTBo0KCsXLmyzfkrVqxIXV1duwLuAbt27cqLL76YT3ziE3/23JaWllx00UW577772uz/53/+55xwwgnp2bNnBg8enM2bN6exsbH1+IE5Bw4c2O65AAAAAAA6goqWeti4cWPuvPPODBs2LGPGjMkbb7zReuyYY47JtddemxEjRmTWrFkZMWJE/sf/+B95+umnM2fOnIqGeumll9LS0pIPf/jDhzy+c+fOvP3226mpqUlVVVWGDRuW73//+znllFNyxhlnZPny5ZkzZ04mT56cJDn77LNTV1eXcePG5fbbb89bb72VKVOm5PLLL8+JJ55Y0WwAAAAAAEe6isLvokWL8vbbb2fx4sVZvHhxm2MjRozIjBkzMnv27NTX1+eR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3/72t3nqqafyqU99KskfAvL+/fuzaNGiXHLJJRk6dGhuvvnmbNmypZKPDwAAAADQIVQUfv/YCy+8kEmTJuXiiy/O0KFDs2fPnjZROEnr9r59+yq696ZNm7J48eJ87Wtfq+i6N954I1/5ylfSq1ev1mtffvnlJEnXrl1z7733Zvr06XnllVfyxS9+MXv27Kno/gAAAAAAR7qKf9ztgCVLluSmm25KXV1dZs2alSSprq4+KPAe2O7atWtF9//pT3+aXr165aKLLmr3Na+88kq++tWvZv/+/Zk7d2569OiRJLn88stz/vnnt1ku4rTTTsv555+fpUuXtq4NDAAAAABQgr/ojd958+Zl7NixufDCC3P//fenuro6SdK3b9+Dlk/YsmVLunXrlu7du1f0jCVLluSyyy5Lp07tG3H16tX567/+63Tt2jU/+tGP8sEPfrDN8T9eI7h379457rjjsnnz5ormAgAAAAA40lUcfufPn59p06Zl5MiRaWhoaLO0w6BBg7Jy5co2569YsSJ1dXXtDrhJsmvXrrz44ov5xCc+0a7z16xZk9GjR+e0007LD3/4w5x44oltjt9999255JJLWn9gLkl+85vfZNu2bTn11FPbPRcAAAAAQEdQUfjduHFj7rzzzgwbNixjxozJG2+8kaampjQ1NWXnzp259tprs2bNmsyaNSsbNmzIQw89lKeffjqjR4+uaKiXXnopLS0t+fCHP3zI4zt37szWrVuTJO+8805uuumm9OrVKzNmzMjevXtbZzpwzrBhw/Lqq6/m9ttvz8aNG/P8889n7Nixqaury3nnnVfRbAAAAAAAR7qK1vhdtGhR3n777SxevDiLFy9uc2zEiBGZMWNGZs+enfr6+jzyyCPp379/6uvrM2TIkIqGOrBcxHHHHXfI49OnT8/KlSuzdOnSrFmzJo2NjUly0HrAH/jAB7J06dKcccYZefDBB3PvvffmiiuuSJcuXfKZz3wmt9xyS6qqqiqaDQAAAADgSFfV8u71D6jI/v3N2br194d7DACAdmls3JipUyfnttum56STTj7c48Bh528CAOhoamrel6OOat8iDn/Rj7sBAAAAAHDkEn4BAAAAAAoj/AIAAAAAFEb4BQAAAAAojPALAAAAAFAY4RcAAAAAoDDCLwAAAABAYYRfAAAAAIDCCL8AAAAAAIURfgEAAAAACiP8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19fnmWeeyYQJE1JTU5MhQ4a03qO5uTn33HNPHn/88YwYMaLN/bdt25ZrrrkmZ599dp544ols2bIlt9xyS5qbm3P77bcfcqZnn322zfbvfve7XHPNNbngggvSr1+/JMm6devyoQ99KPPnz29z7rHHHlvJxwcAAAAA6BDaHX4bGxvz3HPPZf78+Rk4cGCS5NZbb82yZcuycOHCvPnmmxkwYEDGjRuXJKmtrc3atWszZ86c1vC7YcOGTJ48OY2Nja1R9t3mzZuXzp075+677051dXVOPfXU3HDDDXnsscfS0tKSqqqqg6454YQT2mxPmzYtPXv2zLRp01r3vfzyyzn11FMPOhcAAAAAoETtXuO3Z8+eeeCBB3LmmWe27quqqkpVVVV27NiRVatWtXmzN0nOPffcrF69Oi0tLUmSFStWpLa2Nk8++WT69+9/0DOeffbZDBs2LNXV1a37rrzyyvzkJz85ZPQ91PX/8A//kGnTpqVLly6t+9etW5fa2tr2flQAAAAAgA6t3eG3R48eueCCC9oE1UWLFqWxsTHnnXdeNm/enD59+rS5pnfv3tm9e3e2bduWJBk5cmSmT5+eXr16HfIZGzduTO/evfPtb387Q4cOzbBhw3LXXXdl79697ZqxoaEhn/nMZ1rXHD7gV7/6VV555ZVcccUV+eQnP5nrrrsua9asae9HBwAAAADoUNodfv/YCy+8kEmTJuXiiy/O0KFDs2fPnjZROEnr9r59+9p1z127duXBBx/M3r17c9999+Xmm2/OwoUL861vfevPXvv888/nX//1X/P1r3+9zf7XXnstO3fuzFtvvZVvfetbmT17do4//vhcc801Wb9+fTs/LQAAAABAx1HRj7sdsGTJktx0002pq6vLrFmzkiTV1dUHBd4D2127dm3fMJ075+STT279Ibczzjgj+/fvzze+8Y1MnDjxPd8UTpIFCxbkrLPOykc+8pE2+/v27Zvnn38+Xbt2zdFHH50kOfPMM7N27do8+uijmTp1artmAwAAAADoKCp+43fevHkZO3ZsLrzwwtx///2t6/H27ds3W7ZsaXPuli1b0q1bt3Tv3r1d9+7Tp09OO+20NvsObL/66qvveV1zc3OWLl2az33uc4c83qNHj9bomySdOnVKbW1tXn/99XbNBQAAAADQkVQUfufPn59p06Zl5MiRaWhoaLO0w6BB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30HPmDdvXjp37py777471dXVOfXUU3PDDTfkscceS0tLS6qqqg66Zv369Wlubs4555yTY4899qDjv/zlL7Ny5cr87Gc/S21tbZLkjjvuyOjRozN+/PiceOKJlXwNAAAAAABHtIqWeujZs2ceeOCBnHnmma37qqqqUlVVlR07dmTVqlVt3uxNknPPPTerV69OS0tLkmTFihWpra3Nk08+mf79+x/0jGeffTbDhg1LdXV1674rr7wyP/nJTw4ZfZNk3bp1Of744w8ZfZNk1apVOeGEE1qjb5J87GMfS1VVVVavXt3+LwAAAAAAoAOoKPz26NEjF1xwQbp06dK6b9GiRWlsbMx5552XzZs3p0+fPm2u6d27d3bv3p1t27YlSUaOHJnp06enV69eh3zGxo0b07t373z729/O0KFDM2zYsNx1113Zu3fve861bt26dOvWLTfccEM+9alP5XOf+1wefvjhNDc3J0lef/319O3bt801Xbp0yXHHHZfXXnutkq8AAAAAAOCIV/GPu73bCy+8kEmTJuXiiy/O0KFDs2fPnjZROEnr9r59+9p1z127duXBBx/M3r17c9999+Xmm2/OwoUL861vfes9r/nVr36VHTt25JJLLsn3v//9fOELX8i9996bv//7v0+S7N69+6C5kqS6uvpPBmUAAAAAgI6o4h93O2DJkiW56aabUldXl1mzZiX5Q0j948B7YLtr167tG6hz55x88smtP+R2xhlnZP/+/fnGN76RiRMnHvJN4QOhuHv37kmSAQMGZNeuXfnud7+bsWPH5phjjjlkeN67d2+6devW7s8MAAAAANAR/EVv/M6bNy9jx47NhRdemPvvv791Pd6+fftmy5Ytbc7dsmVLunXr1hpl/5w+ffrktNNOa7PvwParr756yGu6dOly0P1PP/30vPXWW/nd736XPn36HDTXvn37sn379vTu3btdcwEAAAAAdBQVh9/58+dn2rRpGTlyZBoaGtosoTBo0KCsXLmyzfkrVqxIXV1dOnVq36MGDx6cNWvWtP4YXJK8/PLLOeqoow75Y3AtLS256KKLct9997XZ/8///M854YQT0rNnzwwePDibN29OY2Nj6/EDcw4cOLBdcwEAAAAAdBQVhd+NGzfmzjvvzLBhwzJmzJi88cYbaWpqSlNTU3bu3Jlrr702a9asyaxZs7Jhw4Y89NBD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VFVVZceOHVm1alWbN3uT5Nxzz83q1avT0tKSJFmxYkVqa2vz5JNPpn///gc949lnn82wYcNSXV3duu/KK6/MT37yk0NG3+bm5sycOfOgN4c7deqUHTt2JEn27duXX//61znllFMq+bgAAAAAAB1SReG3R48eueCCC9KlS5fWfYsWLUpjY2POO++8bN68OX369GlzTe/evbN79+5s27YtSTJy5MhMnz49vXr1OuQzNm7cmN69e+fb3/52hg4dmmHDhuWuu+7K3r17D/0BOnXKkCFD2jz3t7/9bZ566ql86lOfSpKsX78++/fvz6JFi3LJJZdk6NChufnmm7Nly5ZKPj4AAAAAQIdQ8Y+7vdsLL7yQSZMm5eKLL87QoUOzZ8+eNlE4Sev2vn372nXPXbt25cEHH8zevXtz33335eabb87ChQvzrW99q13Xv/HGG/nKV76SXr165Wtf+1qSPyzzkCRdu3bNvffem+nTp+eVV17JF7/4xezZs6e9HxcAAAAAoEOo+MfdDliyZEluuumm1NXVZdasWUmS6urqgwLvge2uXbu2b6DOnXPyySe3/pDbGWeckf379+cb3/hGJk6c+J5vCifJK6+8kq9+9avZv39/5s6dmx49eiRJLr/88px//vmpqalpPfe0007L+eefn6VLl7auDQwAAAAAUIK/6I3fefPmZezYsbnwwgtz//33t67H27dv34OWT9iyZUu6deuW7t27t+veffr0yWmnndZm34HtV1999T2vW716df76r/86Xbt2zY9+9KN88IMfbHP83dE3+cMSFMcdd1w2b97crrkAAAAAADqKisPv/PnzM23atIwcOTINDQ1tlnYYNGhQVq5c2eb8FStWpK6uLp06te9RgwcPzpo1a1p/DC75w1INRx111CF/DC5J1qxZk9GjR+e0007LD3/4w5x44oltjt9999255JJL2tzzN7/5TbZt25ZTTz21XXMBAAAAAHQUFYXfjRs35s4778ywYcMyZsyYvPHGG2lqakpTU1N27tyZa6+9NmvWrMmsWbOyYcOGPPTQQ3n66aczevTodj9j1KhR2bRpU2677bZs3Lgxy5Yty8yZMzN8+PDWt3Z37tyZrVu3Jkneeeed3HTTTenVq1dmzJiRvXv3ts504Jxhw4bl1Vdfze23356NGzfm+eefz9ixY1NXV5fzzjuvkq8AAAAA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",
      "text/plain": [
       "<Figure size 1700x1500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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AAAAAAB0CA3dtg8AAAAAsK4QTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAA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HXdcfA4qAMCH0csvv1jvvPNOs2es81588YVWv9M8PXr0qH79+jd7BgBAm+vU0tLSsronv/fee7X99tvX0KFD6/TTT69evXrVbbfdVldddVVdddVVNWLEiIYHrFixshYuXNLw+wAAmuHll1+sM844pdkzoMOZOPEiARUA+EDYaKOe1aXL6t2Q39CVp127dq05c+ZUly5dqkePHlVVtc0229STTz5ZV1xxxRrFUwCAD5LfXXH6jW+Mqc0227zJa1i6dEmtv37PZs9Ypy1Y8EJNn36Zq7EBgA+lhm/b79nz/f84/cQnPlGzZs1qk0EAAB8Em222eQ0YMKjZMwAAgHbU0A+MevLJJ2uHHXaoOXPmtDr+yCOP1NChQ9t0GAAAAABAMzUUT4cMGVKDBw+uc845px544IF6+umna+LEifXQQw/Vcccd114bAQAAAADWuoZu2+/cuXP96Ec/qosuuqhOOumkevvtt+tTn/pUXXXVVbXlllu210YAAAAAgLWu4WeefuQjH6mJEye2xxYAAAAAgA6jodv2AQAAAADWFeIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAgngIAAAAABOIpAAAAAEAg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X1533nln3Xnnna1eO/DAA2vSpEltOg4AAAAAoFkaiqfHHntsHXvsse21BQAAAACgw2j4macAAAAAAOsC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAAAC8RQAAAAAIBBPAQAAAACCPyueXn755XXYYYe11RYAAAAAgA5jjePpNddcUxdffHEbTgEAAAAA6Di6NvqGl19+ucaNG1dz5sypgQMHtsMkAAAAAIDma/jK01//+tfVrVu3uuWWW2q77bZrj00AAAAAAE3X8JWne++9d+29997tsQUAgLVk/vzna9Git5s948/y4osLaunSpfX000/V0qVLmz1njfXuvUF97GMfb/YMAACChuMpAAAfbG+88UZ9+cv71cqVK5s9pU2MHfv3zZ7wZ+nSpUvdddes6tu3b7OnAADwB8RTAIB1TN++feuWW+74wF95WlW1ZMmS6tmzZ7Nn/Fl6995AOAUA6KDEUwCAdZDbxAEA4P/W8A+MAgAAAABYF4inAAAAAACBeAoAAAAAEHRqaWlpaeaAFStW1sKFS5o5AQAAAABYR2y0Uc/q0mX1ril15SkAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQCCeAgAAAAAE4ikAAAAAQNCppaWlpZkDWlpaauXKpk4AAAAAANYRnTt3qk6dOq3WuU2PpwAAAAAAHZHb9gEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEAAAAAAvEUAAAAACAQTwEA1hFbbbVV3XTTTc2e8SfNmTOnttpqq5o/f367fsZhhx1Wp59++hr/HX/og/C9BQCgcV2bPQAAAD7oZs2aVb179272DAAA2ph4CgAAf6ZNNtmk2RMAAGgHbtsHAFiHzJ07t4444ogaNmxYff7zn6/LL7+81et33313HXLIITV8+PDabbfdauLEifXOO++sej3dnv77x5YtW1ZnnXVW7brrrjVs2LA64IAD6j/+4z9WndvS0lLTp0+vffbZp7bbbrv6yle+Urfccsv7dt5zzz01cuTI2mabbWr//fevu+++e9VrK1asqBkzZtR+++1Xw4YNq/3226+uvfbaP/o1v/vuuzVhwoQaMWJE7bjjjnXhhRfWypUrG/q+Pfvss3XkkUfWjjvuWMOHD68jjzyyHn/88fd9D+bPn19bbbVV/PWv//qvVVW1aNGiOvvss2uXXXapHXfcsQ4//PD61a9+1dAeAADWDvEUAGAd8pOf/KQOOOCAuv322+vQQw+tyZMn1+zZs6uq6s4776zjjjuu9txzz7rpppvqH/7hH+r222+vsWPHrvbnX3LJJfX444/XtGnT6vbbb6/dd9+9Tj755FXPH/3+979f1157bZ199tl166231uGHH17jx4+va665ptXnXH311avOGThwYJ100km1ZMmSqqqaNGlSXXbZZfXNb36zbr311ho9enSdf/75NWPGjLjpvPPOq9tvv70mTZpU//zP/1wvvfRSPfDAAw1938aOHVv9+vWrG2+8sW644Ybq3LlzffOb33zfef37969Zs2at+vWzn/2sdtppp9pyyy1r3333rZaWlvrGN75Rzz//fF1++eV1/fXX1/bbb1+HHnpoPfroow1tAgCg/bltHwBgHfK3f/u3dcABB1RV1ZgxY+rKK6+sRx55pEaMGFHTpk2rfffdt8aMGVNVVYMGDaqWlpY6/vjj66mnnqqhQ4f+n5//3HPPVc+ePevjH/94bbDBBnXiiSfWZz7zmerTp08tXbq0ZsyYUZMnT64999yzqqq22GKLeuGFF+qKK66o0aNHr/qcM888sz772c9WVdXxxx9fd911Vz399NM1ePDguvbaa+v000+vUaNGVVXVwIEDa/78+TVt2rT6u7/7u1Z7Fi9eXDfddFONGzeu9thjj6qqmjBhQt13330Nfd+ee+65+tznPlebb755devWrSZMmFDPPPNMrVy5sjp3/v/XI3Tp0qXVLfznn39+Pfvss3X99ddXr169avbs2fXQQw/VfffdVxtuuGFV/W+YffDBB+vqq6+uSZMmNbQLAID2JZ4CAKxDBg4c2OrPG2ywQf32t7+tqqonnnii9t9//1av77zzzqteW514+o1vfKOOPfbYGjFiRG277ba166671qhRo6p37971P//zP/Xb3/62TjnllFbB8b333qt333231eMBBg0a1GpjVdU777xTzzzzTC1fvrx23HHH9+38p3/6p3r99ddbHZ87d24tX768hg0bturYeuutV5/61Kf+z6/l95188sk1YcKEmjlzZu288871+c9/vkaOHNnq6/hD11xzTV133XV19dVX1+abb15VVb/+9a+rpaWl9tprr1bnvvvuu6v+PwAA0HGIpwAA65AuXbq871hLS0ur33/f754N2rVr/mfje++91+rPw4cPr3vuuafuvffemj17dt188801derU+vGPf1zrr79+VVVdfPHFNXjw4Pd9Vvfu3Vf9d4qSLS0tceOf2tmpU6f4tf2xr+ePGT16dH3xi1+se+65p2bPnl1TpkypqVOn1s0331wf+chH3nf+z372s5owYUJNmjSptt9++1Y7e/Xq9b7nxla1/voBAOgYPPMUAICq+t8fevTggw+2Ova7Z4MOGTKkqqq6detWixcvXvX6vHnzWp0/ZcqU+sUvflH77LNPfec736k77rijPv7xj9cdd9xRgwcPrq5du9aCBQtqwIABq37dc889dcUVV/zJqzh/Z8iQIdWtW7f6xS9+8b6dm2yySfXp06fV8UGDBtV6663X6ut677336rHHHluN78j/ev311+ucc86p5cuX10EHHVQXXnhh3XLLLfXqq6/W/fff/77zH3/88Tr55JPr6KOPXvVogd/Zcssta/HixbV8+fJW34Pp06fXT3/609XeBADA2uHKUwAAqqrqqKOOqhNPPLEuu+yy+tKXvlTPPvtsnXvuubXXXnutiqfbb7993XDDDfWZz3ymWlpaauLEia2umHz++efrlltuqXPPPbe22GKLevjhh2vBggU1fPjw6t27d/3N3/xNXXLJJdWrV6/aYYcdas6cOXXhhRfWMcccs1obe/XqVV/96ldrypQpteGGG9awYcNq1qxZNXPmzBo7duyqK01/p2fPnvW1r32tpkyZUptsskkNGTKkrrzyynr55ZdX+/vSp0+fuvvuu+u5556rU045ZdWVo926dattttmm1bmvvvrqqscWHHbYYfXqq6+ueq1Hjx71+c9/vrbeeus6+eST66yzzqr+/fvXzJkz66abbqorrrhitTcBALB2iKcAAFRV1X777VeTJ0+uqVOn1mWXXVYbbbRRjRw5sk444YRV54wfP77Gjx9fhxxySG266aZ14okn1ksvvbTq9XHjxtUFF1xQ3/rWt+rNN9+szTffvE499dT6yle+UlVVZ5xxRvXt27cuueSSeuWVV6p///51wgkn1FFHHbXaO3/3Gd/73vfqtddeq4EDB9Z3v/vdOuSQQ+L5p5xySq233np1zjnn1JIlS+pLX/pS7b333qv993Xt2rWmT59eF1xwQR1xxBG1bNmy2nrrrWvatGm1xRZbtDr3v//7v2vBggW1YMGCuvPOO1u9duCBB9akSZPqyiuvrAsvvLBOOumkWrZsWQ0ZMqR+8IMf1IgRI1Z7EwAAa0enlj/24CgAAAAAgHWYZ54CAAAAAARu2wcAYJ01ffr0uuyyy/7kOWeeeWYdfPDBa2kRAAAdidv2AQBYZ7311lv15ptv/slzNt544+rVq9faGQQAQIcingIAAAAABJ55CgAAAAAQiKcAAAAAAIF4CgAAAAAQiKcAAAAAAIF4CgAAAAAQiKcAAAAAAIF4CgAAAAAQiKcAAAAAAMH/A29e00kzJMEnAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1700x1500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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FQAAAABAgbgKAAAAAFAgrgIAAAAAFIirAAAAAAAF4ioAAAAAQIG4CgAAAABQIK4CAAAAABSIqwAAAAAABeIqAAAAAECBuAoAAAAAUCCuAgAAAAAUiKsAAAAAAAXiKgAAAABAgbgKAAAAAFAgrgIAAAAAFIirAAAAAAAF4ioAAAAAQIG4CgAAAABQIK4CAAAAABSIqwAAAAAABeIqAAAAAECBuAoAAAAAUCCuAgAAAAAUiKsAAAAAAAXiKgAAAABAgbgKAAAAAFAgrgIAAAAAFIirAAAAAAAF4ioAAAAAQIG4CgAAAABQIK4CAAAAABSIqwAAAAAABeIqAAAAAECBuAoAAAAAUCCuAgAAAAAUiKsAAAAAAAXiKgAAAABAgbgKAAAAAFAgrgIAAAAAFIirAAAAAAAF4ioAAAAAQIG4CgAAAABQIK4CAAAAABSIqwAAAAAABeIqAAAAAECBuAoAAAAAUCCuAgAAAAAUiKsAAAAAAAXiKgAAAABAgbgKAAAAAFAgrgIAAAAAFIirAAAAAAAF4ioAAAAAQIG4CgAAAABQIK4CAAAAABSIqwAAAAAABeIqAAAAAECBuAoAAAAAUCCuAgAAAAAUiKsAAAAAAAXiKgAAAABAgbgKAAAAAFAgrgIAAAAAFIirAAAAAAAF4ioAAAAAQIG4CgAAAABQIK4CAAAAABSIqwAAAAAABeIqAAAAAECBuAoAAAAAUCCuAgAAAAAUiKsAAAAAAAXiKgAAAABAgbgKAAAAAFAgrgIAAAAAFIirAAAAAAAF4ioAAAAAQIG4CgAAAABQIK4CAAAAABSIqwAAAAAABeIqAAAAAECBuAoAAAAAUCCuAgAAAAAUiKsAAAAAAAXiKgAAAABAgbgKAAAAAFAgrgIAAAAAFIirAAAAAAAF4ioAAAAAQIG4CgAAAABQIK4CAAAAABSIqwAAAAAABeIqAAAAAECBuAoAAAAAUCCuAgAAAAAUiKsAAAAAAAXiKgAAAABAgbgKAAAAAFAgrgIAAAAAFIirAAAAAAAF4ioAAAAAQIG4CgAAAABQIK4CAAAAABSIqwAAAAAABeIqAAAAAECBuAoAAAAAUCCuAgAAAAAUiKsAAAAAAAXiKgAAAABAgbgKAAAAAFAgrgIAAAAAFIirAAAAAAAF4ioAAAAAQIG4CgAAAABQIK4CAAAAABSIqwAAAAAABeIq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AAAAAAAF4ioAAAAAQIG4CgAAAABQIK4CAAAAABSIqwAAAAAABeIqAAAAAECBuAoAAAAAUCCuAgAAAAAUiKsAAAAAAAVtjquPP/54vva1r2XYsGEZMmRIjjjiiNx///0t5++55558+tOfzk477ZThw4dn6tSp7TowAAAAAEBH0Oa4evTRR2f+/Pm56KKL8sMf/jBNTU353Oc+l+eeey5PPPFERo0alS233DLTpk3L0UcfnbPOOivTpk17LWYHAAAAAGiYLm25+Kmnnsrmm2+eMWPGZNCgQUmSo446Kh/5yEdy33335bbbbsuGG26YCRMmpEuXLhk4cGBLiB05cuRr8gYAAAAAABqhTStXe/XqlbPPPrslrC5ZsiRTpkxJ//79s80222TmzJnZdddd06XL35rtsGHDMm/evDz22GPtOzkAAAAAQAO1aeXqC5188sm55ppr0rVr11x44YXp3r17Fi1a1BJe1+rbt2+SZOHChdl4443XbVoAAAAAgA6izXuurvXZz34206ZNy4c//OEcffTRufvuu7N8+fJ07dq11XXdunVLkqxYsWLdJgUAAAAA6EDKK1e32WabJMmpp56a3//+97niiivS1NSUlStXtrpubVTt3r37OowJAAAAANCxtGnl6pIlS3LTTTfl+eef/9sTbLBBttlmmyxevDj9+/fP4sWLW92z9nG/fv3aYVwAAAAAgI6hTXH1scceywknnJDbbrut5diqVasya9asDBw4MEOHDs0dd9yR1atXt5yfMWNGBgwYkD59+rTf1AAAAAAADdamuDpo0KDstdde+eY3v5nf/va3mTNnTk466aQsXbo0n/vc5zJy5MgsW7Ys48aNy9y5czN9+vRMmTIlY8aMea3mBwAAAABoiDZ/odXEiROz22675fjjj8/BBx+cJ598Mv/5n/+ZzTbbLH369MnkyZPzwAMPZMSIETn//PMzduzYjBgx4rWYHQAAAACgYdr8hVY9e/bMKaecklNOOeUlz++44465+uqr13UuAAAAAIAOrc0rVwEAAAAAEFcBAAAAAErEVQAAAACAAnEVAAAAAKBAXAUAAAAAKBBXAQAAAAAKxFUAAAAAgAJxFQAAAACgQFwFAAAAACgQVwEAAAAACsRVAAAAAIACcRUAAAAAoEBcBQAAAAAoEFcBAAAAAArEVQAAAACAAnEVAAAAAKBAXAUA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AAAKxFUAAAAAgAJxFQAAAACgQFwFAAAAACgQVwEAAAAACsRVAAAAAIACcRUAAAAAoEBcBQAAAAAoEFcBAAAAAArEVQAAAACAAnEVAAAAAKCgU3Nzc3Ojh3g5zc3NWbOmQ48IAAAAAKxHNtigUzp16vSK13X4uAoAAAAA0BHZFgAAAAAAoEBcBQAAAAAoEFcBAAAAAArEVQAAAACAAnEVAAAAAKBAXAUAAAAAKBBXAQAAAAAKxFUAAAAAgAJxFQAAAACgQFwFAAAAACgQVwEAAAAACsRVAAAAAIACcRUAgIYbP358hgwZkp133jmPPfZYo8f5pxs+fHjOO++8dn3OX/ziF5k7d267PicAAK2JqwAANNTs2bNz5ZVX5sQTT8yPfvSjbLzxxo0e6XXv4YcfzpFHHpnHH3+80aMAAKzXxFUAABpq6dKlSZLdd989W2yxRYOnWT80Nzc3egQAgDcEcRUAYD02Z86cjBkzJkOHDs0OO+yQ/fbbL5deemnL+RtuuCEf+MAHMnjw4Bx88MGZOnVqtt1225bzTz/9dE4++eQMGzYsO++8cw477LD88Y9/bNMMTz75ZMaPH5+99947O+64Y/71X/81v/nNb5Ik06dPz2c+85kkyf7775+TTjrpVT3nZz7zmZx88sk5+OCDs8suu+T6669PkkybNi0f+MAHsuOOO+YDH/hALr/88qxZs6blvuuuuy4f+tCHMnjw4Oy555459dRTs3LlypZZ9tprr1xzzTXZY489MmTIkBx99NF59NFHW+5fvnx5zjnnnOy3334ZPHhwPvKRj+SnP/1py/np06fngAMOaPnvDjvskI997GO54447Wv1NTzzxxOyyyy4ZNmxYLrvsshe9vzvvvDOHHnpodtxxx+yzzz4ZP358li1b1nJ++PDhueSSS3LMMcdkyJAhee9735tvfvObef7557NgwYLst99+SZLDDjus3bcbAADgb8RVAID11HPPPZfRo0dno402ylVXXZUbb7wxBx54YM4444zcc889+cUvfpETTzwxH//4x3P99dfnYx/7WM4666yW+5ubm/OFL3whDz30UL73ve/lmmuuyU477ZRDDjkks2bNelUzrF69OqNHj87MmTNz5plnZvr06Rk0aFAOP/zw/OEPf8gHP/jBlvh37bXXZty4ca/6/V177bU57LDDcuWVV2bPPffM1VdfnW9/+9v50pe+lJtuuinHHXdcLr744pb3NHv27Pzbv/1bjjnmmPz0pz/Nt771rfzoRz/K5MmTW55zyZIlufzyy3POOefk8ssvz8KFC/P5z38+zz//fJLkhBNOyHXXXZeTTz45119/ffbff/98+ctfzs0339zyHAsXLsxVV12VM888M//1X/+VN73pTTnppJNaVpMed9xx+cMf/pBJkyblsssuyy9/+cs8/PDDLffPnj07o0aNyp577pnrr78+Z511Vu6+++6MHj261YrUc889N0OHDs3111+fsWPH5oorrsiNN96YTTfdNNdee22S5Lzzzsvo0aNf9d8UAIC26dLoAQAAeG0899xzOeyww3LooYemR48eSZJjjz02kydPzr333psf/vCHOfDAA3P44YcnSQYMGJB58+ZlypQpSZIZM2bkrrvuyowZM7LRRhsl+WtcvPPOOzN16tScfvrprzjDrbfemrvvvjs33HBDBg0alOSvX171xz/+MZdccknOPffc9OrVK0nSu3fv9OzZ81W/v+233z4HHXRQy+Pvfve7+eIXv5gPfehDSZK3ve1tWbZsWcaPH58vf/nLWbBgQTp16pTNN988m222WTbbbLNccsklefOb39zyHKtWrcoZZ5yRHXbYIUly5pln5oMf/GBuu+22bLbZZrnlllsyadKk7LPPPkmSY445JrNnz86kSZOy//77tzzH+PHjs/322ydJRo0alaOPPjp/+ctfsmzZstx6662ZMmVKdtlllyTJ2WefnX333bdlhksuuSS77757jjzyyCTJ1ltvnbPPPjv7779/br/99rz3ve9Nkuyxxx457LDDWt7r97///dx555356Ec/mt69eydJevXq1fL/HgCA9ieuAgCsp3r37p1PfepTufHGGzNr1qw8+OCDmT17dpJkzZo1ufvuu/P+97+/1T1Dhw5tiat33313mpubW4W/JFm5cmVWrFjxqmaYM2dOevbs2RJWk6RTp07ZZZddcuutt67Du0u22mqrlt+XLFmSRYsWZeLEiTn33HNbjq9ZsyYrVqzIggULsueee2bIkCH5+Mc/ni222CK777579ttvv5aQmiQ9evRo9XjgwIHp1atX5syZk6effjpJsvPOO7eaY+jQoZk4cWKrYwMHDmz5fW0wXrVqVebMmZMkGTx4cMv5jTfeOG9729taHs+aNSvz58/PkCFDXvSe77///pa4+sLXWPs6q1atesm/FQAArw1xFQBgPfWXv/wln/zkJ9O7d+8MHz48e+yxRwYPHpy99947SdKlS5dW+5H+vTVr1uTNb35zpk+f/qJzXbt2fVUz/KMvVmpubk6XLuv2T9GmpqaW39e+j69//et53/ve96JrN91003Tt2jVTp07NrFmzcuutt+bWW2/NkUcemY9+9KM57bTTkiQbbrjhi+5dvXp1Onfu/A/neKn38lJ/n+bm5nTq1KnVvGu98P41a9bkoIMOalm5+kJrV6S+3GsAAPDPY89VAID11I033pgnn3wyP/jBD3LUUUflgAMOyFNPPZXkrxFuu+22y+9///tW9/zud79r+X3QoEFZtmxZVq1ala222qrl5+KLL84tt9zyqmbYdttt8/TTT7es2Fz72nfccUe22WabdniXf9WnT5/07t07Dz30UKtZ77777pxzzjlJkl/96lc5//zz8853vjNHHHFEpk6dmmOPPTY//vGPW57nySefzEMPPdTy+L777suyZcvyzne+s+WLvl745VRJMnPmzFf9XtZuFXDnnXe2HFu6dGkefPDBlsfveMc7Mnfu3Fbv4/nnn89pp52WhQsXvqrXWRtxAQB4bYmrAADrqf79++e5557LT37ykzzyyCO59dZbc8IJJyT560f7v/CFL+QnP/lJLrvsssybNy/Tpk3LFVdc0XL/nnvume233z7HH398ZsyYkfnz5+e0007L9OnTX/SR9H9kjz32yPbbb5+vfOUruf3223P//fdnwoQJmTNnTj772c+223vt1KlTvvCFL+T73/9+rrjiijz44IP52c9+llNOOSVNTU3p2rVrNtxww1xwwQWZMmVKHnroofzpT3/KL3/5yxd9/P5rX/ta/vSnP+Wuu+7K2LFjM2TIkAwdOjQDBw7Mvvvum/Hjx+eXv/xlHnjggZx//vm55ZZbXvWXRm255ZY58MADM2HChPzf//1f5syZk7Fjx2blypUt14wePTqzZs3K+PHjc//99+d3v/tdvvKVr2TevHnZeuutX9XrdO/ePUlabWcAAED7sy0AAMB66sADD8zdd9+d008/PcuWLcvmm2+egw8+OLfcckv++Mc/5pBDDsmECRPyve99L2effXZ22GGHHHLIIS2BtXPnzrn00ktz5pln5rjjjstzzz2XgQMH5vzzz89uu+32qmZY+xxnnHFGvvSlL2XlypXZYYcdMmXKlOy0007t+n5Hjx6dbt265fvf/35OP/30bLzxxvnEJz6RY489Nknyvve9L6eeemouvfTSfOc730lTU1P23nvvnHTSSa2e56CDDsoRRxyRlStXZvjw4Rk3blzLStCJEydm4sSJGTduXJYuXZpBgwblvPPOywEHHPCq5zzjjDNyxhln5Pjjj8+aNWvyyU9+Mkv+/3buUDW5OIwD8LtgEcvQok2vwOpViIh94MKCWZl4BbYVEbTY5VyBawbdHRhXBItoEdv3tUU5HBgyfJ54zvt/efMv/I7Hn//1ej3m83l8fHxEq9WKfD4fjUYjBoNB6jqG5+fnaLfbMR6P4/v7O0ajUer7AABI7+mfYiYAgIf09fUVpVIparXaz7fpdBrL5TJWq9UdL7uPJEni/f09drvdvU8BAOCPUAsAAPCg1ut1dLvd2Gw2sd/v4/PzMxaLRTSbzXufBgAAf4JaAACAB9Xr9eJyuUS/34/j8RjlcjleXl7i9fU11fu3t7fYbrc3Z5IkiWq1mvqm2WwWk8nk5sxwOIxOp5N6JwAA/Ba1AAAAZHI4HOJ6vd6cqVQqkcvlUu88n89xOp1uzhSLxSgUCql3AgDAbxGuAgAAAABkoHMVAAAAACAD4SoAAAAAQAbCVQAAAACADISrAAAAAAAZCFcBAAAAADIQrgIAAAAAZCBcBQAAAADIQLgKAAAAAJDBf+/3xzzsC4bnAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1700x1500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "for feat in num_features:\n",
    "  plot_boxplot(data, feat)\n",
    "  print(\"\\n\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "country                   0\n",
       "year                      0\n",
       "uniqueid                  0\n",
       "bank_account              0\n",
       "location_type             0\n",
       "cellphone_access          0\n",
       "household_size            0\n",
       "age_of_respondent         0\n",
       "gender_of_respondent      0\n",
       "relationship_with_head    0\n",
       "marital_status            0\n",
       "education_level           0\n",
       "job_type                  0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 164,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Analyzing the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: xlabel='gender_of_respondent', ylabel='count'>"
      ]
     },
     "execution_count": 165,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sb.countplot(x='gender_of_respondent',\n",
    "             data=data,\n",
    "             hue='bank_account',\n",
    "             palette=\"GnBu_d\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [],
   "source": [
    "sb.countplot(x='marital_status',\n",
    "             data=data,\n",
    "             hue='bank_account',\n",
    "             palette=\"GnBu_d\")\n",
    "sb.set(rc={'figure.figsize':(17,15)})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {},
   "outputs": [],
   "source": [
    "sb.countplot(x='education_level',\n",
    "             data=data,\n",
    "             hue='bank_account',\n",
    "             palette=\"GnBu_d\")\n",
    "sb.set(rc={'figure.figsize':(17,15)})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [],
   "source": [
    "for x in cat_features:\n",
    "    data[x].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {},
   "outputs": [],
   "source": [
    "#changing the categorical values to numerical values\n",
    "x = pd.Categorical(data['education_level'])               # Male=1,Female=0\n",
    "data['education_level']=x.codes\n",
    "\n",
    "x = pd.Categorical(data['marital_status'])               # Male=1,Female=0\n",
    "data['marital_status']=x.codes\n",
    "\n",
    "x = pd.Categorical(data['gender_of_respondent'])               # Male=1,Female=0\n",
    "data['gender_of_respondent']=x.codes\n",
    "\n",
    "x = pd.Categorical(data['relationship_with_head'])               # Male=1,Female=0\n",
    "data['relationship_with_head']=x.codes\n",
    "\n",
    "x = pd.Categorical(data['job_type'])               # Male=1,Female=0\n",
    "data['job_type']=x.codes\n",
    "\n",
    "x = pd.Categorical(data['location_type'])               # Male=1,Female=0\n",
    "data['location_type']=x.codes\n",
    "\n",
    "x = pd.Categorical(data['bank_account'])               # Male=1,Female=0\n",
    "data['bank_account']=x.codes\n",
    "\n",
    "x = pd.Categorical(data['cellphone_access'])               # Male=1,Female=0\n",
    "data['cellphone_access']=x.codes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "country                   0\n",
       "year                      0\n",
       "uniqueid                  0\n",
       "bank_account              0\n",
       "location_type             0\n",
       "cellphone_access          0\n",
       "household_size            0\n",
       "age_of_respondent         0\n",
       "gender_of_respondent      0\n",
       "relationship_with_head    0\n",
       "marital_status            0\n",
       "education_level           0\n",
       "job_type                  0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 170,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "metadata": {},
   "outputs": [],
   "source": [
    "drop_features = ['year','country', 'uniqueid','relationship_with_head','marital_status','household_size']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "metadata": {},
   "outputs": [],
   "source": [
    "new_data = data.drop(columns = drop_features, axis = 1 )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "bank_account            0\n",
       "location_type           0\n",
       "cellphone_access        0\n",
       "age_of_respondent       0\n",
       "gender_of_respondent    0\n",
       "education_level         0\n",
       "job_type                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 195,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_data.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>bank_account</th>\n",
       "      <th>location_type</th>\n",
       "      <th>cellphone_access</th>\n",
       "      <th>age_of_respondent</th>\n",
       "      <th>gender_of_respondent</th>\n",
       "      <th>education_level</th>\n",
       "      <th>job_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>24</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>70</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>26</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>34</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>26</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   bank_account  location_type  cellphone_access  age_of_respondent  \\\n",
       "0             1              0                 1                 24   \n",
       "1             0              0                 0                 70   \n",
       "2             1              1                 1                 26   \n",
       "3             0              0                 1                 34   \n",
       "4             0              1                 0                 26   \n",
       "\n",
       "   gender_of_respondent  education_level  job_type  \n",
       "0                     0                3         9  \n",
       "1                     0                0         4  \n",
       "2                     1                5         9  \n",
       "3                     0                2         3  \n",
       "4                     1                2         5  "
      ]
     },
     "execution_count": 187,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>bank_account</th>\n",
       "      <th>location_type</th>\n",
       "      <th>cellphone_access</th>\n",
       "      <th>age_of_respondent</th>\n",
       "      <th>gender_of_respondent</th>\n",
       "      <th>education_level</th>\n",
       "      <th>job_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>-0.907879</td>\n",
       "      <td>0</td>\n",
       "      <td>0.827143</td>\n",
       "      <td>1.158412</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.002712</td>\n",
       "      <td>0</td>\n",
       "      <td>-1.674274</td>\n",
       "      <td>-0.460314</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>-0.781332</td>\n",
       "      <td>1</td>\n",
       "      <td>2.494754</td>\n",
       "      <td>1.158412</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>-0.275142</td>\n",
       "      <td>0</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>-0.784060</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>-0.781332</td>\n",
       "      <td>1</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>-0.136569</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   bank_account  location_type  cellphone_access  age_of_respondent  \\\n",
       "0             1              0                 1          -0.907879   \n",
       "1             0              0                 0           2.002712   \n",
       "2             1              1                 1          -0.781332   \n",
       "3             0              0                 1          -0.275142   \n",
       "4             0              1                 0          -0.781332   \n",
       "\n",
       "   gender_of_respondent  education_level  job_type  \n",
       "0                     0         0.827143  1.158412  \n",
       "1                     0        -1.674274 -0.460314  \n",
       "2                     1         2.494754  1.158412  \n",
       "3                     0        -0.006663 -0.784060  \n",
       "4                     1        -0.006663 -0.136569  "
      ]
     },
     "execution_count": 188,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scaler = StandardScaler()\n",
    "new_data[['age_of_respondent','education_level','job_type']]=pd.DataFrame(scaler.fit_transform(new_data[['age_of_respondent','education_level','job_type']], ))\n",
    "new_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>bank_account</th>\n",
       "      <th>location_type</th>\n",
       "      <th>cellphone_access</th>\n",
       "      <th>age_of_respondent</th>\n",
       "      <th>gender_of_respondent</th>\n",
       "      <th>education_level</th>\n",
       "      <th>job_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>22902.000000</td>\n",
       "      <td>22902.000000</td>\n",
       "      <td>22902.000000</td>\n",
       "      <td>22330.000000</td>\n",
       "      <td>22902.000000</td>\n",
       "      <td>22330.000000</td>\n",
       "      <td>22330.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.141953</td>\n",
       "      <td>0.393634</td>\n",
       "      <td>0.746922</td>\n",
       "      <td>-0.000158</td>\n",
       "      <td>0.410314</td>\n",
       "      <td>0.000021</td>\n",
       "      <td>-0.002069</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.349009</td>\n",
       "      <td>0.488566</td>\n",
       "      <td>0.434785</td>\n",
       "      <td>1.000072</td>\n",
       "      <td>0.491901</td>\n",
       "      <td>0.999785</td>\n",
       "      <td>1.000606</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-1.414069</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-1.674274</td>\n",
       "      <td>-1.755295</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-0.781332</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>-1.107805</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.211868</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>-0.136569</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.610690</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.827143</td>\n",
       "      <td>1.158412</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.825271</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.494754</td>\n",
       "      <td>1.158412</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       bank_account  location_type  cellphone_access  age_of_respondent  \\\n",
       "count  22902.000000   22902.000000      22902.000000       22330.000000   \n",
       "mean       0.141953       0.393634          0.746922          -0.000158   \n",
       "std        0.349009       0.488566          0.434785           1.000072   \n",
       "min        0.000000       0.000000          0.000000          -1.414069   \n",
       "25%        0.000000       0.000000          0.000000          -0.781332   \n",
       "50%        0.000000       0.000000          1.000000          -0.211868   \n",
       "75%        0.000000       1.000000          1.000000           0.610690   \n",
       "max        1.000000       1.000000          1.000000           2.825271   \n",
       "\n",
       "       gender_of_respondent  education_level      job_type  \n",
       "count          22902.000000     22330.000000  22330.000000  \n",
       "mean               0.410314         0.000021     -0.002069  \n",
       "std                0.491901         0.999785      1.000606  \n",
       "min                0.000000        -1.674274     -1.755295  \n",
       "25%                0.000000        -0.006663     -1.107805  \n",
       "50%                0.000000        -0.006663     -0.136569  \n",
       "75%                1.000000         0.827143      1.158412  \n",
       "max                1.000000         2.494754      1.158412  "
      ]
     },
     "execution_count": 189,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "bank_account            0\n",
       "location_type           0\n",
       "cellphone_access        0\n",
       "age_of_respondent       0\n",
       "gender_of_respondent    0\n",
       "education_level         0\n",
       "job_type                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 196,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_data.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = new_data.drop(columns=['bank_account'], axis=1)\n",
    "Y = new_data['bank_account']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location_type</th>\n",
       "      <th>cellphone_access</th>\n",
       "      <th>age_of_respondent</th>\n",
       "      <th>gender_of_respondent</th>\n",
       "      <th>education_level</th>\n",
       "      <th>job_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>24</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>70</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>26</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>34</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>26</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   location_type  cellphone_access  age_of_respondent  gender_of_respondent  \\\n",
       "0              0                 1                 24                     0   \n",
       "1              0                 0                 70                     0   \n",
       "2              1                 1                 26                     1   \n",
       "3              0                 1                 34                     0   \n",
       "4              1                 0                 26                     1   \n",
       "\n",
       "   education_level  job_type  \n",
       "0                3         9  \n",
       "1                0         4  \n",
       "2                5         9  \n",
       "3                2         3  \n",
       "4                2         5  "
      ]
     },
     "execution_count": 198,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location_type</th>\n",
       "      <th>cellphone_access</th>\n",
       "      <th>age_of_respondent</th>\n",
       "      <th>gender_of_respondent</th>\n",
       "      <th>education_level</th>\n",
       "      <th>job_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>2.290200e+04</td>\n",
       "      <td>2.290200e+04</td>\n",
       "      <td>2.290200e+04</td>\n",
       "      <td>2.290200e+04</td>\n",
       "      <td>2.290200e+04</td>\n",
       "      <td>2.290200e+04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>-9.928114e-17</td>\n",
       "      <td>6.949680e-17</td>\n",
       "      <td>6.949680e-17</td>\n",
       "      <td>4.095347e-17</td>\n",
       "      <td>1.489217e-17</td>\n",
       "      <td>-7.942492e-17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.000022e+00</td>\n",
       "      <td>1.000022e+00</td>\n",
       "      <td>1.000022e+00</td>\n",
       "      <td>1.000022e+00</td>\n",
       "      <td>1.000022e+00</td>\n",
       "      <td>1.000022e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>-8.057098e-01</td>\n",
       "      <td>-1.717948e+00</td>\n",
       "      <td>-1.414069e+00</td>\n",
       "      <td>-8.341561e-01</td>\n",
       "      <td>-1.674274e+00</td>\n",
       "      <td>-1.755295e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>-8.057098e-01</td>\n",
       "      <td>-1.717948e+00</td>\n",
       "      <td>-7.813320e-01</td>\n",
       "      <td>-8.341561e-01</td>\n",
       "      <td>-6.662580e-03</td>\n",
       "      <td>-1.107805e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>-8.057098e-01</td>\n",
       "      <td>5.820898e-01</td>\n",
       "      <td>-2.118684e-01</td>\n",
       "      <td>-8.341561e-01</td>\n",
       "      <td>-6.662580e-03</td>\n",
       "      <td>-1.365690e-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.241142e+00</td>\n",
       "      <td>5.820898e-01</td>\n",
       "      <td>6.106901e-01</td>\n",
       "      <td>1.198816e+00</td>\n",
       "      <td>8.271429e-01</td>\n",
       "      <td>1.158412e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.241142e+00</td>\n",
       "      <td>5.820898e-01</td>\n",
       "      <td>2.825271e+00</td>\n",
       "      <td>1.198816e+00</td>\n",
       "      <td>2.494754e+00</td>\n",
       "      <td>1.158412e+00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       location_type  cellphone_access  age_of_respondent  \\\n",
       "count   2.290200e+04      2.290200e+04       2.290200e+04   \n",
       "mean   -9.928114e-17      6.949680e-17       6.949680e-17   \n",
       "std     1.000022e+00      1.000022e+00       1.000022e+00   \n",
       "min    -8.057098e-01     -1.717948e+00      -1.414069e+00   \n",
       "25%    -8.057098e-01     -1.717948e+00      -7.813320e-01   \n",
       "50%    -8.057098e-01      5.820898e-01      -2.118684e-01   \n",
       "75%     1.241142e+00      5.820898e-01       6.106901e-01   \n",
       "max     1.241142e+00      5.820898e-01       2.825271e+00   \n",
       "\n",
       "       gender_of_respondent  education_level      job_type  \n",
       "count          2.290200e+04     2.290200e+04  2.290200e+04  \n",
       "mean           4.095347e-17     1.489217e-17 -7.942492e-17  \n",
       "std            1.000022e+00     1.000022e+00  1.000022e+00  \n",
       "min           -8.341561e-01    -1.674274e+00 -1.755295e+00  \n",
       "25%           -8.341561e-01    -6.662580e-03 -1.107805e+00  \n",
       "50%           -8.341561e-01    -6.662580e-03 -1.365690e-01  \n",
       "75%            1.198816e+00     8.271429e-01  1.158412e+00  \n",
       "max            1.198816e+00     2.494754e+00  1.158412e+00  "
      ]
     },
     "execution_count": 199,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scaler = StandardScaler()\n",
    "X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)\n",
    "X.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 200,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "location_type\n",
      "cellphone_access\n",
      "age_of_respondent\n",
      "gender_of_respondent\n",
      "education_level\n",
      "job_type\n"
     ]
    }
   ],
   "source": [
    "for col in X.columns:\n",
    "  print(col)      "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location_type           0\n",
       "cellphone_access        0\n",
       "age_of_respondent       0\n",
       "gender_of_respondent    0\n",
       "education_level         0\n",
       "job_type                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 201,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       -0.907879\n",
       "1        2.002712\n",
       "2       -0.781332\n",
       "3       -0.275142\n",
       "4       -0.781332\n",
       "           ...   \n",
       "22897   -1.160974\n",
       "22898    0.610690\n",
       "22899   -0.718058\n",
       "22900   -0.718058\n",
       "22901   -0.528237\n",
       "Name: age_of_respondent, Length: 22902, dtype: float64"
      ]
     },
     "execution_count": 202,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X['age_of_respondent']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# For All the models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 203,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a 70 - 30 split of your data for the train and validation/test dataset\n",
    "xtrain, xval_test, ytrain, yval_test = train_test_split(X, Y, test_size=0.3, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "metadata": {},
   "outputs": [],
   "source": [
    "xval, xtest, yval, ytest = train_test_split(xval_test, yval_test, test_size=0.3, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((16031, 6), (16031,))"
      ]
     },
     "execution_count": 205,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xtrain.shape, ytrain.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((4809, 6), (4809,))"
      ]
     },
     "execution_count": 206,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xval.shape, yval.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((2062, 6), (2062,))"
      ]
     },
     "execution_count": 207,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xtest.shape, ytest.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a dictionary to store the names of the model and the actual model definition (or creation)\n",
    "models = {'randomforest':RandomForestClassifier(n_estimators=400,max_depth=5),\n",
    "        'gradient':GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=2),\n",
    "        'tree':DecisionTreeClassifier(max_depth=3),\n",
    "        'logistic':LogisticRegression(),\n",
    "        'neighbors':KNeighborsClassifier(n_neighbors=3)}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 209,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_val_models(dict_, xtrain,ytrain, xval, yval, xtest, ytest):\n",
    "\n",
    "  train_scores = [] #A list to store all the training evaluation scores\n",
    "  val_scores = [] #A list to store all the validation evaluation scores\n",
    "  test_scores = [] #A list to store all the test evaluation scores\n",
    "\n",
    "  for name, model in dict_.items():\n",
    "    model.fit( xtrain, ytrain )\n",
    "    ytrain_pred = model.predict(xtrain)\n",
    "    yval_pred = model.predict(xval)\n",
    "    ytest_pred = model.predict(xtest)\n",
    "    train_score = metrics.f1_score(ytrain, ytrain_pred)\n",
    "    val_score = metrics.f1_score(yval, yval_pred)\n",
    "    test_score = metrics.f1_score(ytest, ytest_pred)\n",
    "    train_scores.append(train_score)\n",
    "    val_scores.append(val_score)\n",
    "    test_scores.append(test_score)\n",
    "\n",
    "\n",
    "  score_df = pd.DataFrame([train_scores,val_scores,test_scores], columns = list(dict_.keys())).transpose()\n",
    "  score_df.rename(columns = {0:'Train', 1:'Valid', 2:'Test'}, inplace = True)\n",
    "\n",
    "  return score_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 210,
   "metadata": {},
   "outputs": [],
   "source": [
    "score_df = train_val_models(models, xtrain,ytrain, xval, yval, xtest, ytest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 211,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Train</th>\n",
       "      <th>Valid</th>\n",
       "      <th>Test</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>randomforest</th>\n",
       "      <td>0.326761</td>\n",
       "      <td>0.355456</td>\n",
       "      <td>0.351220</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>gradient</th>\n",
       "      <td>0.407804</td>\n",
       "      <td>0.414359</td>\n",
       "      <td>0.410480</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>tree</th>\n",
       "      <td>0.254777</td>\n",
       "      <td>0.310263</td>\n",
       "      <td>0.322251</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>logistic</th>\n",
       "      <td>0.329692</td>\n",
       "      <td>0.345200</td>\n",
       "      <td>0.345455</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>neighbors</th>\n",
       "      <td>0.567645</td>\n",
       "      <td>0.430901</td>\n",
       "      <td>0.425287</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Train     Valid      Test\n",
       "randomforest  0.326761  0.355456  0.351220\n",
       "gradient      0.407804  0.414359  0.410480\n",
       "tree          0.254777  0.310263  0.322251\n",
       "logistic      0.329692  0.345200  0.345455\n",
       "neighbors     0.567645  0.430901  0.425287"
      ]
     },
     "execution_count": 211,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Implementing cross val"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 215,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define a function to check the performace of different models \n",
    "def cross_val_models(model_dict, X, Y):\n",
    "\n",
    "  cv_scores = [] #A list to store the mean performance of each model\n",
    "\n",
    "  for name, model in model_dict.items():\n",
    "    cv_score = cross_val_score(model, X, Y, cv=10, scoring='f1').mean()\n",
    "    cv_scores.append(cv_score)\n",
    "\n",
    "  score_df = pd.DataFrame([cv_scores], columns = list(model_dict.keys())).transpose()\n",
    "  score_df.rename(columns = {0:'CV_Score'}, inplace = True)\n",
    "\n",
    "  return score_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 216,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CV_Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>randomforest</th>\n",
       "      <td>0.322256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>gradient</th>\n",
       "      <td>0.381820</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>tree</th>\n",
       "      <td>0.278751</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>logistic</th>\n",
       "      <td>0.315320</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>neighbors</th>\n",
       "      <td>0.401486</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              CV_Score\n",
       "randomforest  0.322256\n",
       "gradient      0.381820\n",
       "tree          0.278751\n",
       "logistic      0.315320\n",
       "neighbors     0.401486"
      ]
     },
     "execution_count": 216,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score_df = cross_val_models(models, X, Y)\n",
    "score_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "# using hyper parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import RandomizedSearchCV, GridSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 226,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a dictionary to store the names of the model and the actual model definition (or creation)\n",
    "models = {'random forest':RandomForestClassifier(random_state=42),\n",
    "          'gradient boosting': GradientBoostingClassifier(random_state=42)\n",
    "          }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 220,
   "metadata": {},
   "outputs": [],
   "source": [
    "params = {'random forest': \n",
    "          {\n",
    "          'n_estimators': range(40, 101, 10),\n",
    "          'max_depth': range(2,11,1),                  #'max_depth': [4,5,6,7,8,9]\n",
    "          'min_samples_leaf': range(2,9,2),           #'min_samples_leaf': [2,3,4,5,6,7]\n",
    "          },\n",
    "\n",
    "          'gradient boosting': \n",
    "          {\n",
    "          'max_depth':range(2,11,1),\n",
    "          'min_samples_leaf':range(2,9,2),\n",
    "          'n_estimators':range(40, 100, 10)\n",
    "          }\n",
    "         }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 229,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define a function to check the performace of different models \n",
    "def hyper_parameters(model_dict,param_dict, X, Y):\n",
    "\n",
    "  best_scores = [] #A list to store the mean performance of each model\n",
    "  best_params = {}\n",
    "\n",
    "  for name, model in model_dict.items():\n",
    "    # grid = GridSearchCV(model, param_dict[name], cv=10, scoring='r2')\n",
    "    grid = RandomizedSearchCV(model, param_dict[name], cv=10, scoring='f1')\n",
    "    grid.fit(X,Y)\n",
    "    best_score = grid.best_score_\n",
    "    best_param = grid.best_params_\n",
    "    best_scores.append(best_score)\n",
    "    best_params[name] = best_param\n",
    "\n",
    "  score_df = pd.DataFrame([best_scores], columns = list(model_dict.keys())).transpose()\n",
    "  score_df.rename(columns = {0:'Best_Score'}, inplace = True)\n",
    "\n",
    "  return score_df,best_params"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 230,
   "metadata": {},
   "outputs": [],
   "source": [
    "score_df,best_params = hyper_parameters(models,params, X, Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 232,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Best_Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>random forest</th>\n",
       "      <td>0.408756</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>gradient boosting</th>\n",
       "      <td>0.423459</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   Best_Score\n",
       "random forest        0.408756\n",
       "gradient boosting    0.423459"
      ]
     },
     "execution_count": 232,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 233,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>random forest</th>\n",
       "      <th>gradient boosting</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>n_estimators</th>\n",
       "      <td>70</td>\n",
       "      <td>80</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min_samples_leaf</th>\n",
       "      <td>2</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max_depth</th>\n",
       "      <td>10</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  random forest  gradient boosting\n",
       "n_estimators                 70                 80\n",
       "min_samples_leaf              2                  6\n",
       "max_depth                    10                  8"
      ]
     },
     "execution_count": 233,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.DataFrame(best_params)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 234,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location_type</th>\n",
       "      <th>cellphone_access</th>\n",
       "      <th>age_of_respondent</th>\n",
       "      <th>gender_of_respondent</th>\n",
       "      <th>education_level</th>\n",
       "      <th>job_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.907879</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>0.827143</td>\n",
       "      <td>1.158412</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>-1.717948</td>\n",
       "      <td>2.002712</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>-1.674274</td>\n",
       "      <td>-0.460314</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.241142</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.781332</td>\n",
       "      <td>1.198816</td>\n",
       "      <td>2.494754</td>\n",
       "      <td>1.158412</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.275142</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>-0.784060</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.241142</td>\n",
       "      <td>-1.717948</td>\n",
       "      <td>-0.781332</td>\n",
       "      <td>1.198816</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>-0.136569</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22897</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-1.160974</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>0.187176</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22898</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>0.610690</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>-1.674274</td>\n",
       "      <td>0.510922</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22899</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.718058</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>0.827143</td>\n",
       "      <td>0.510922</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22900</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.718058</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>0.510922</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22901</th>\n",
       "      <td>1.241142</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.528237</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>0.827143</td>\n",
       "      <td>1.158412</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>22902 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       location_type  cellphone_access  age_of_respondent  \\\n",
       "0          -0.805710          0.582090          -0.907879   \n",
       "1          -0.805710         -1.717948           2.002712   \n",
       "2           1.241142          0.582090          -0.781332   \n",
       "3          -0.805710          0.582090          -0.275142   \n",
       "4           1.241142         -1.717948          -0.781332   \n",
       "...              ...               ...                ...   \n",
       "22897      -0.805710          0.582090          -1.160974   \n",
       "22898      -0.805710          0.582090           0.610690   \n",
       "22899      -0.805710          0.582090          -0.718058   \n",
       "22900      -0.805710          0.582090          -0.718058   \n",
       "22901       1.241142          0.582090          -0.528237   \n",
       "\n",
       "       gender_of_respondent  education_level  job_type  \n",
       "0                 -0.834156         0.827143  1.158412  \n",
       "1                 -0.834156        -1.674274 -0.460314  \n",
       "2                  1.198816         2.494754  1.158412  \n",
       "3                 -0.834156        -0.006663 -0.784060  \n",
       "4                  1.198816        -0.006663 -0.136569  \n",
       "...                     ...              ...       ...  \n",
       "22897             -0.834156        -0.006663  0.187176  \n",
       "22898             -0.834156        -1.674274  0.510922  \n",
       "22899             -0.834156         0.827143  0.510922  \n",
       "22900             -0.834156        -0.006663  0.510922  \n",
       "22901             -0.834156         0.827143  1.158412  \n",
       "\n",
       "[22902 rows x 6 columns]"
      ]
     },
     "execution_count": 234,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 241,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a dictionary to store the names of the model and the actual model definition (or creation)\n",
    "models_ = {'random forest':RandomForestClassifier(n_estimators = 40,min_samples_leaf=8, max_depth = 7,random_state=42),\n",
    "          'gradient boosting': GradientBoostingClassifier(n_estimators = 70,min_samples_leaf=4, max_depth = 4, random_state=42 ),\n",
    "          'support vector': SVC(),\n",
    "          'logistic regression': LogisticRegression()}\n",
    "\n",
    "#Define a function to check the performace of different models \n",
    "def cross_val_models(model_dict, X, Y):\n",
    "\n",
    "  cv_scores = [] #A list to store the mean performance of each model\n",
    "\n",
    "  for name, model in model_dict.items():\n",
    "    cv_score = cross_val_score(model, X, Y, cv=10, scoring='f1').mean()\n",
    "    cv_scores.append(cv_score)\n",
    "\n",
    "  score_df = pd.DataFrame([cv_scores], columns = list(model_dict.keys())).transpose()\n",
    "  score_df.rename(columns = {0:'CV_Score'}, inplace = True)\n",
    "\n",
    "  return score_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 242,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CV_Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>random forest</th>\n",
       "      <td>0.359746</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>gradient boosting</th>\n",
       "      <td>0.403802</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>support vector</th>\n",
       "      <td>0.333797</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>logistic regression</th>\n",
       "      <td>0.315320</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     CV_Score\n",
       "random forest        0.359746\n",
       "gradient boosting    0.403802\n",
       "support vector       0.333797\n",
       "logistic regression  0.315320"
      ]
     },
     "execution_count": 242,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score_df = cross_val_models(models_, X, Y)\n",
    "score_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Deployment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 243,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = GradientBoostingClassifier(n_estimators = 70,min_samples_leaf=4, max_depth = 4, random_state=42 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 246,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location_type</th>\n",
       "      <th>cellphone_access</th>\n",
       "      <th>age_of_respondent</th>\n",
       "      <th>gender_of_respondent</th>\n",
       "      <th>education_level</th>\n",
       "      <th>job_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.907879</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>0.827143</td>\n",
       "      <td>1.158412</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>-1.717948</td>\n",
       "      <td>2.002712</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>-1.674274</td>\n",
       "      <td>-0.460314</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.241142</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.781332</td>\n",
       "      <td>1.198816</td>\n",
       "      <td>2.494754</td>\n",
       "      <td>1.158412</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.275142</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>-0.784060</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.241142</td>\n",
       "      <td>-1.717948</td>\n",
       "      <td>-0.781332</td>\n",
       "      <td>1.198816</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>-0.136569</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22897</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-1.160974</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>0.187176</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22898</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>0.610690</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>-1.674274</td>\n",
       "      <td>0.510922</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22899</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.718058</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>0.827143</td>\n",
       "      <td>0.510922</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22900</th>\n",
       "      <td>-0.805710</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.718058</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>-0.006663</td>\n",
       "      <td>0.510922</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22901</th>\n",
       "      <td>1.241142</td>\n",
       "      <td>0.582090</td>\n",
       "      <td>-0.528237</td>\n",
       "      <td>-0.834156</td>\n",
       "      <td>0.827143</td>\n",
       "      <td>1.158412</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>22902 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       location_type  cellphone_access  age_of_respondent  \\\n",
       "0          -0.805710          0.582090          -0.907879   \n",
       "1          -0.805710         -1.717948           2.002712   \n",
       "2           1.241142          0.582090          -0.781332   \n",
       "3          -0.805710          0.582090          -0.275142   \n",
       "4           1.241142         -1.717948          -0.781332   \n",
       "...              ...               ...                ...   \n",
       "22897      -0.805710          0.582090          -1.160974   \n",
       "22898      -0.805710          0.582090           0.610690   \n",
       "22899      -0.805710          0.582090          -0.718058   \n",
       "22900      -0.805710          0.582090          -0.718058   \n",
       "22901       1.241142          0.582090          -0.528237   \n",
       "\n",
       "       gender_of_respondent  education_level  job_type  \n",
       "0                 -0.834156         0.827143  1.158412  \n",
       "1                 -0.834156        -1.674274 -0.460314  \n",
       "2                  1.198816         2.494754  1.158412  \n",
       "3                 -0.834156        -0.006663 -0.784060  \n",
       "4                  1.198816        -0.006663 -0.136569  \n",
       "...                     ...              ...       ...  \n",
       "22897             -0.834156        -0.006663  0.187176  \n",
       "22898             -0.834156        -1.674274  0.510922  \n",
       "22899             -0.834156         0.827143  0.510922  \n",
       "22900             -0.834156        -0.006663  0.510922  \n",
       "22901             -0.834156         0.827143  1.158412  \n",
       "\n",
       "[22902 rows x 6 columns]"
      ]
     },
     "execution_count": 246,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 247,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GradientBoostingClassifier(max_depth=4, min_samples_leaf=4, n_estimators=70,\n",
       "                           random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GradientBoostingClassifier</label><div class=\"sk-toggleable__content\"><pre>GradientBoostingClassifier(max_depth=4, min_samples_leaf=4, n_estimators=70,\n",
       "                           random_state=42)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "GradientBoostingClassifier(max_depth=4, min_samples_leaf=4, n_estimators=70,\n",
       "                           random_state=42)"
      ]
     },
     "execution_count": 247,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X,Y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# saving the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 248,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 249,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(\"big_mart_model.pkl\", \"wb\") as f:\n",
    "  pickle.dump(model,f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 252,
   "metadata": {},
   "outputs": [],
   "source": [
    "# loading the saved model\n",
    "loaded_model = pickle.load(open('C:/Users/Al Amin/Downloads/NITDA_AI_Class-main/NITDA_AI_Class-main/ML_Compete/Classification/big_mart_model.pkl', 'rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}

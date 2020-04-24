﻿import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

integer_features = ['Occupation_Satisfaction', 
                    'Yearly_IncomeK',
                    'Last_school_grades',
                    'Number_of_differnt_parties_voted_for',
                    'Number_of_valued_Kneset_members',
                    'Num_of_kids_born_last_10_years'
                    ]
float_features = ['Avg_monthly_expense_when_under_age_21', 
                  'Avg_lottary_expanses',
                  'Avg_monthly_expense_on_pets_or_plants',
                  'Avg_environmental_importance',
                  'Financial_balance_score_(0-1)',
                  '%Of_Household_Income',
                  'Avg_size_per_room',
                  'Garden_sqr_meter_per_person_in_residancy_area',
                  'Avg_Residancy_Altitude',
                  'Yearly_ExpensesK',
                  '%Time_invested_in_work',
                  'Avg_education_importance',
                  'Avg_Satisfaction_with_previous_vote',
                  'Avg_monthly_household_cost',
                  'Phone_minutes_10_years',
                  'Avg_government_satisfaction',
                  'Weighted_education_rank',
                  '%_satisfaction_financial_policy',
                  'Avg_monthly_income_all_years',
                  'Political_interest_Total_Score',
                  'Overall_happiness_score'

                  ]
nominal_features = ['Age_group',
                    'Looking_at_poles_results',
                    'Married',
                    'Gender',
                    'Voting_Time',
                    'Will_vote_only_large_party',
                    'Most_Important_Issue',
                    'Main_transportation',
                    'Occupation',
                    'Financial_agenda_matters'
                   ]


def train_test_split(x, test_size, validation_size):
    """
    :param x: features
    :param y: labels
    :param test_size: test set size in percentage (0 < test_size < 1)
    :param validation_size: validation set size in percentage (0 < validation_size < 1)
    :return X_train, X_test, y_train, y_test
    """
    n_samples = x.shape[0]
    num_train = int((1 - test_size - validation_size) * n_samples)
    num_test = int(test_size * n_samples)
    rand_gen = np.random.RandomState()

    x_train = []
    x_test = []
    x_validation = []

    indexes = rand_gen.permutation(n_samples)

    for i in range(n_samples):
        if i < num_train:
            x_train.append(x.iloc[indexes[i]])
        elif ((i >= num_train) & (i < num_train + num_test)):
            x_test.append(x.iloc[indexes[i]])
        else:
            x_validation.append(x.iloc[indexes[i]])
    return pd.DataFrame.from_records(x_train), pd.DataFrame.from_records(x_test), pd.DataFrame.from_records(
        x_validation)




def main():
    df = pd.read_csv("ElectionsData.csv")

    # split the data to train , test and validation
    df_train, df_test, df_validation = train_test_split(df, 0.2, 0.2)

    # Save the raw data first
    df_train.to_csv('raw_train.csv', index=False)
    df_test.to_csv('raw_test.csv', index=False)
    df_validation.to_csv('raw_validation.csv', index=False)
    
    # Convert nominal types to numerical categories
    # from nominal to Categorical
    df = df.apply(lambda x:  pd.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    # give number to each Categorical
    df = df.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)

    # 1 - Imputation - Complete missing values
    
    # Fill missing values with the most common value per column
    df.loc[:, nominal_features] = df.apply(lambda x: x.fillna(x.mode()[0]), axis=0)
    
    # Fill by mean rounded to nearest integer
    df.loc[:, integer_features] = df.apply(lambda x: x.fillna(round(x.mean())), axis=0)
    
    # Fill by mean
    df.loc[:, float_features] = df.apply(lambda x: x.fillna(x.mean()), axis=0)


    # 2 - Data Cleansing
    # Outlier detection using z score
    z = np.abs(stats.zscore(df))
    df = df[(z < 3).all(axis=1)]

    # Remove lines with wrong party (Violets | Khakis)
    df = df[df.Vote != 10]
    df = df[df.Vote != 4]


    # 3 - Normalization (scaling)


    # 4 - Feature Selection


    # Save the transformed data
    df_train = df.iloc[0:round(len(df)*0.6), :]
    df_test = df.iloc[round(len(df)*0.6)+1:round(len(df)*0.8)+1, :]
    df_validation = df.iloc[round(len(df)*0.8)+1:len(df)+1, :]

    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    df_validation.to_csv('validation.csv', index=False)

    


if __name__ == "__main__":
    main()
import pandas
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

def main():
    df = pandas.read_csv("ElectionsData.csv")

    # Save the raw data first
    df_train = df.iloc[0:6000, :]
    df_test = df.iloc[6001:8001, :]
    df_validation = df.iloc[8001:10001, :]
    df_train.to_csv('raw_train.csv', index=False)
    df_test.to_csv('raw_test.csv', index=False)
    df_validation.to_csv('raw_validation.csv', index=False)
    
    # Convert nominal types to numerical categories
    df = df.apply(lambda x:  pandas.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    df = df.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)

    # 1 - Imputation
    
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
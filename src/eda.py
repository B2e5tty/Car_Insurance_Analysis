import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, fisher_exact


class EDA:
    # initialize the class and variable
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    # funtion to return sepecific columns
    def return_columns(self,col):
        return self.df[col]

    # overview of the dataset
    def overview(self):
        print("Shape of the dataframe:")
        print(f"{self.df.shape}\n")

        print("Datatypes of each columns:")
        print(f"{self.df.dtypes}\n")

        print("Information on the data:")
        print(f"{self.df.info()}\n")

        print("Describe the numerical column statistics:")
        print(f"{self.df.describe()}\n")

        print("The first five rows of the data:")
        print(f"{self.df.head(5)}\n")

        print("The bottom five rows of the date:")
        print(f"{self.df.tail(5)}\n")

    # function for variability of numerical columns
    def variance_of_columns(self):
        columns = self.df.select_dtypes(np.float64).columns
        name = []
        df_min = []            # minimum value  
        df_max = []            # maximum valuess
        df_mean = []           # mean of the columns
        df_std = []            # standard deviation of the columns   
        df_iqr = []            # interquatile of the columns

        for col in columns:
            name.append(col)
            df_min.append(self.df[col].min())
            df_max.append(self.df[col].max())
            df_mean.append(self.df[col].mean())
            df_std.append(self.df[col].std())
            df_iqr.append((self.df[col].quantile(0.75) - self.df[col].quantile(0.25)))

        df_dict = {'Featues':name, 'Minimum':df_min, 'Maximum':df_max,'Mean':df_mean, 'Standard Deviation':df_std, 'IQR': df_iqr}

        df = pd.DataFrame(df_dict)

        return df 

    # function to return columns with null values
    def columns_with_null(self):
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                print(col)

    # function to change datatype of columns
    def change_dtype(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object' and col != 'TransactionMonth':
                self.df[col] = self.df[col].astype('category')

            elif col == 'TransactionMonth':
                self.df[col] = pd.to_datetime(self.df[col])

            elif self.df[col].dtype == 'int64' and col != 'PolicyID':
                self.df[col] = self.df[col].astype('category')

            elif col == 'PolicyID':
                self.df[col] = self.df[col].astype('object')

            else:
                pass

        # Return the columns with their changed dtypes
        print(self.df.info())

    # function to remove null values
    def remove_missing_value(self):
        self.df.dropna(inplace = True)

        # check 
        self.df.info()

    # function to remove outlier outside 95% of data distribution
    def fix_outlier(self):
        column = self.df.select_dtypes(np.float64).columns.tolist()
        column.remove('TotalClaims')

        for col in column:
            self.df[col] = np.where(self.df[col] > self.df[col].quantile(0.95), self.df[col].median(), self.df[col])
       
        return self
    
    # function for boxplot
    def box_plot(self):
        column = self.df.select_dtypes(np.float64).columns

        fig, axs = plt.subplots(2, 2, figsize=(12,7))
        axs = axs.flatten()

        for i,col in enumerate(column):
            sns.boxplot(data = self.df, x = col, ax = axs[i])
            axs[i].set_xlabel(col)
            axs[i].set_ylabel('Distribution')
            axs[i].set_xscale('log')

        plt.tight_layout()
        plt.show()

    # function to plot histogram
    def histogram(self):
        columns = self.df.select_dtypes(np.number)

        fig, axs = plt.subplots(2, 2, figsize=(12,7))
        axs = axs.flatten()

        for i,col in enumerate(columns):
            sns.histplot(data = self.df, x = col, bins = 80, ax = axs[i])
            axs[i].set_xlabel(col)
            axs[i].set_ylabel('Distribution')
            axs[i].set_xscale('log')

        plt.tight_layout()
        plt.show()

    # function for barplot
    def bar_plot(self):
        columns = ['Gender', 'Province', 'VehicleType', 'RegistrationYear', 'TermFrequency', 'Product', 'StatutoryRiskType', 'Section']

        fig, axs =plt.subplots(4, 2, figsize=(18,13))
        axs = axs.flatten()

        for i,col in enumerate(columns):
            df_value = self.df[col].value_counts()
            axs[i].bar(df_value.index,df_value.values)
            axs[i].set_xlabel(col,fontsize=15)
            axs[i].set_ylabel('Policies',fontsize=15)
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=15)    
  
        plt.tight_layout()
        plt.show()

    # function for scatter plot
    def scatter_plot(self):
        df = self.df.sort_values(by=['PostalCode','TransactionMonth'])
        df['premiumchange'] = df.groupby('PostalCode')['TotalPremium'].pct_change()
        df['claimchange'] = df.groupby('PostalCode')['TotalClaims'].pct_change()

        df_change = df[['TotalPremium','TotalClaims','premiumchange','claimchange']].dropna()
        corr = df_change.corr()
        print('Correlation Matrix between Premium and Claim monthly change:\n')
        print(corr)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='premiumchange', y='claimchange', data=df)
        plt.title('Monthly Changes in TotalPremium vs. TotalClaims by ZipCode')
        plt.xlabel('Monthly Change in TotalPremium')
        plt.ylabel('Monthly Change in TotalClaims')
        plt.show()

    # function fot trend over geography of cover types
    def province_trend_coverType(self):
        province = self.df['Province'].unique()
        cover = self.df['CoverType'].unique()

        for p in province:
            new_df = self.df[self.df['Province'] == p]
            count_cover = new_df['CoverType'].value_counts()


            plt.figure(figsize=(23,9))
            plt.bar(count_cover.index,count_cover.values)
            plt.title(p + ' Cover Types', fontsize=20)
            plt.xlabel('Cover Types', fontsize=18)
            plt.ylabel('Count', fontsize=18)
            plt.xticks(rotation = 45)

    # function for trend over geography for the numerical columns
    def province_trend(self):
        columns = self.df.select_dtypes(np.float64).columns

        for col in columns:
            geo_df = self.df.groupby('Province')[col].mean().reset_index()

            plt.figure(figsize=(17,10))
            sns.barplot(data = geo_df, x = 'Province', y = col)
            plt.title(col + ' trend over province')
            plt.xlabel('Province')
            plt.ylabel('Average Value of ' + col)

        plt.show()

    # return the dataframe
    def get_dataframe(self):
        return self.df


    # Group A and B based on KPI features for risk
    def group_AB_risk(self,col):
        col_grouped = self.df.groupby(col).agg(
            total_claims = ('TotalClaims', 'sum'),
            total_policy = ('PolicyID', 'count')
        )

        col_grouped['risk_score'] = col_grouped['total_claims'] / col_grouped['total_policy']

        col_grouped.sort_values(by='risk_score', inplace=True)

        median_risk = col_grouped['risk_score'].median()

        group_a = self.df[self.df[col].isin(col_grouped[col_grouped['risk_score'] <= median_risk].index)]
        group_b = self.df[self.df[col].isin(col_grouped[col_grouped['risk_score'] > median_risk].index)]

        # undersampling for balancing    
        size_of_smaller_group = min(len(group_a), len(group_b))
        group_a_balanced = group_a.sample(size_of_smaller_group, random_state=42)
        group_b_balanced = group_b.sample(size_of_smaller_group, random_state=42)

        return group_a_balanced, group_b_balanced
    
    # Group A and B based on KPI features for risk
    def group_AB_margin(self,col):
        df = self.df
        df['margin'] = df['TotalPremium'] - df['TotalClaims']

        df.sort_values(by='margin', inplace=True)

        median_profit = df['margin'].median()

        group_a = df[df['margin'] <= median_profit]
        group_b = df[df['margin'] > median_profit]

        # undersampling for balancing    
        size_of_smaller_group = min(len(group_a), len(group_b))
        group_a_balanced = group_a.sample(size_of_smaller_group, random_state=42)
        group_b_balanced = group_b.sample(size_of_smaller_group, random_state=42)

        return group_a_balanced, group_b_balanced
    
    # check if the groups don't have statistics difference for categorical columns
    def chi2_test(self,feature,col,group_a,group_b):
        df_balanced = pd.concat([group_a, group_b])
        df_balanced['group'] = df_balanced[feature].apply(lambda x: 'A' if x in group_a[feature].unique() else 'B')  
        contingency_table = pd.crosstab(df_balanced['group'], df_balanced[col])
        

        if (contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2):
            print(f"Skipping {col} due to insufficient data.")
            return None, None

        else:
            _, p, _, _ = chi2_contingency(contingency_table)

            # To calculate cramers_v
            chi2 = chi2_contingency(contingency_table)[0]
            n = contingency_table.sum().sum()
            r, k = contingency_table.shape
            v = np.sqrt(chi2 / (n * (min(r, k) - 1)))

            return p, v
    
    # check if the groups don't have statistics difference for numerical columns
    def t_test_numerical(self,feature,col,group_a,group_b):
        values_a = self.df[self.df[feature].isin(group_a[feature].unique() )][col]
        values_b = self.df[self.df[feature].isin(group_b[feature].unique() )][col]

        t_stat, p_value = ttest_ind(values_a,values_b)

        # effective size using cohen's d
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        std_a = np.std(values_a, ddof=1)  # Standard deviation for group A
        std_b = np.std(values_b, ddof=1)  # Standard deviation for group B
        n_a = len(values_a)
        n_b = len(values_b)

        # Pooled standard deviation
        s_pooled = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

        # Cohen's d
        cohen_d = (mean_a - mean_b) / s_pooled

        return p_value, cohen_d
    
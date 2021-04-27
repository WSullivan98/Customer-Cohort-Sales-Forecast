import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Transaction Analysis
def sales_per_year(df):
    annual_sales = pd.DataFrame(df.groupby("Year")['Sales Total'].sum())
    return annual_sales


# def transaction_data(df):
#     transaction_data = pd.DataFrame()
#     Number of customers service per year
#     Number of new customers per year
#     Sales per year
#     Orders per year
#     Average sales per order
#     Average # orders per customer
#     Average sales per new customer order
#     Average number of orders per new customer





def sales_per_year_plt(df):
    sales_per_year_plt = df
    sales_per_year_plt['FY'] = ['2015', '2016', '2017', '2018', '2019', '2020']
    fig, ax = plt.subplots()
    x = sales_per_year_plt['FY']
    y = sales_per_year_plt['Sales Total']
    ax.set_title('Sales by Year', fontsize=16)
    ax.set_xlabel('Years', fontsize=16)
    ax.set_ylabel('Sales ($000,000)', fontsize=16)
    plt.yticks(np.arange(0, 15000000, 2000000), labels=['0','2m', '4m', '6m', '8m', '10m', '12m', '14m'], fontsize=16)
    plt.xticks(fontsize=16)
    ax.plot(x,y)
    plt.show() 
    return plt


# Cohort Analysis


def transactions_by_cohort(df):
    Cohort_yr = pd.DataFrame(df.groupby('Customer ID')['Year'].min()).reset_index()
    Cohort_yr.columns = ['Customer ID', 'Cohort Yr']
    cohort_transactions= df.merge(Cohort_yr, on='Customer ID').reset_index(drop=True)
    cohort_transactions = cohort_transactions[['Customer ID', 'Cohort Yr', 'Customer', 'Type', 'Date', 'Year',
                            'Month', 'Document Number', 'Description', 'Qty', 'Unit Price',
                            'Sales Total', 'Revenue', 'State']]
    return cohort_transactions

def cohort_size(df):
    cohort_by_customers = pd.DataFrame(df.groupby('Cohort Yr')['Customer ID'].nunique())
    return cohort_by_customers

def cohort_sales_by_year(df):
    sales_yr_cohort = pd.DataFrame(df.groupby(['Year','Cohort Yr'])['Sales Total'].sum())
    return sales_yr_cohort

def customer_cohort_chart(df):

    sales_yr_cohort = cohort_sales_by_year(df)
    sales_yr_cohort = sales_yr_cohort.reset_index()
    years = df['Year'].unique()

    cohort_sales_data={}
    for y in years:
        cohort_sales_data[y] = sales_yr_cohort[sales_yr_cohort['Cohort Yr']==y]['Sales Total'].to_list()
    print(cohort_sales_data)

    fig, ax = plt.subplots(figsize=(15,12))
    x = df['Year'].unique()

    
    y1 =    cohort_sales_data[y].values() #
    y2 =
    y3 =
    y4 =
    y5 =
    y6 =

    ax.stackplot(years, y1, y2,y3, y4, y5, y6, labels=sales.keys())
    ax.legend(loc='upper left', fontsize=16)
    ax.set_title('Sales by Customer Cohort Year', fontsize=16)
    ax.set_xlabel('Years', fontsize=16)
    ax.set_ylabel('Sales ($000,000)',fontsize=16)
    plt.yticks(np.arange(0, 15000000, 2000000), labels=['0','2m', '4m', '6m', '8m', '10m', '12m', '14m'], fontsize=16)
    plt.xticks(fontsize=16)
    plt.show()
    return plt





if __name__ == "__main__":
    df = pd.read_csv('../data/processed/sales.csv')
    print(df.head())

    cohort_transactions = transactions_by_cohort(df)
    print(cohort_transactions.head())

    cohort_by_customers = cohort_size(df)
    print('Cohort by customers ',cohort_by_customers,'\n')

    sales_yr_cohort = cohort_sales_by_year(df)
    print(sales_yr_cohort)

    annual_sales = sales_per_year(df)
    print('Sales by cohort ',annual_sales.transpose(),'\n' )

    sales_per_year_plt(annual_sales)
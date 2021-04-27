import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
#%inline matplotlib
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")

def combine_dfs(frames):
    combined = pd.concat(frames)
    combined.dropna(axis=1, how='all', inplace=True)
    combined.dropna(axis=0, how='all', inplace=True)
    return combined

def anonymize_customers(df):
    anonymized = pd.DataFrame(df.groupby('Customer')['Date'].min().reset_index())
    anonymized.columns = ['Customer', 'Customer since']
    len_ = len(anonymized['Customer'])
    anonymized['Customer ID'] = np.arange(100001, 100001 + len_,1)
    anonymized.dropna(how='any', inplace=True)
    return anonymized


def merge_dfs(df,lst,key):
    for d in lst:
        df = df.merge(d, on=key)
    return df

def remove_nans(df):
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    df.reset_index(drop=True)
    return df

if __name__ == "__main__":
    sales_2015 = pd.read_csv('../data/raw/Product_sales_by_state_and_customer_2015.csv',parse_dates=['Date'])
    sales_2016 = pd.read_csv('../data/raw/Product_sales_by_state_and_customer_2016.csv',parse_dates=['Date'])
    sales_2017 = pd.read_csv('../data/raw/product_sales_by_state_and_customer_2017.csv',parse_dates=['Date'])
    sales_2018 = pd.read_csv('../data/raw/product_sales_by_state_and_customer_2018.csv',parse_dates=['Date'])
    sales_2019 = pd.read_csv('../data/raw/Product_sales_by_state_and_customer_2019.csv',parse_dates=['Date'])
    sales_2020 = pd.read_csv('../data/raw/Product_sales_by_state_and_customer_2020.csv',parse_dates=['Date'])

    frames = [sales_2020, sales_2019, sales_2018, sales_2017, sales_2016, sales_2015]
    transactions = combine_dfs(frames)
    transactions = transactions.rename({'Address: Billing Address State' :'State',
                                       'Customer/Project: Company Name': 'Customer', 
                                        'Qty. Sold':'Qty',
                                       'Sales Price': 'Unit Price'},
                                         axis=1)
    
    # Date transformations
    transactions['Date'] = pd.to_datetime(transactions['Date'])
    transactions['Year'] = transactions['Date'].dt.year
    transactions['Month'] = transactions['Date'].dt.month
    transactions['Date'].astype(str)

    # Anonymize customers
    customer_keys = anonymize_customers(transactions)
    customer_keys.to_csv('../data/processed/customer_keys.csv')
    transactions = merge_dfs(transactions, [customer_keys], 'Customer')
    transactions = remove_nans(transactions)

    # object to number transformation
    transactions['Revenue'] = transactions['Revenue'].replace(',','').replace('\$|,', '', regex=True)
    transactions['Revenue'] = transactions['Revenue'].astype(str).str.replace("(", "-").str.rstrip(")").astype(float)


    transactions['Unit Price'] = transactions['Unit Price'].replace(',','').replace('\$|,', '', regex=True)
    transactions['Unit Price'] = transactions['Unit Price'].astype(str).str.replace("(", "-").str.rstrip(")").astype(float)
    transactions['Unit Price'] = transactions['Unit Price'].astype(float)

    ## transactions['Unit Price'] = transactions['Unit Price'].replace(',','').replace('\$|,', '', regex=True)
    ## transactions['Unit Price'] = transactions['Unit Price'].astype('float64')

    transactions['Qty'] = transactions['Qty'].replace(',','').replace('\$|,', '', regex=True)
    transactions['Qty'] = transactions['Qty'].astype('float64').astype('int32')

    # object to string transformation
    transactions['Description'] = transactions['Description'].astype(str)
    transactions["State"] = transactions["State"].astype(str)
    transactions['Customer'] = transactions['Customer'].astype(str)
    transactions['Document Number'] = transactions['Document Number'].astype(str)




    transactions['Sales Total'] = transactions['Qty'] * transactions['Unit Price']



    # Add cohorts
    Cohort_yr = pd.DataFrame(transactions.groupby('Customer ID')['Year'].min()).reset_index()
    Cohort_yr.columns = ['Customer ID', 'Cohort Yr']        
    transactions = transactions.merge(Cohort_yr, on='Customer ID').reset_index(drop=True)
    transactions = remove_nans(transactions)

    # reorder columns
    transactions =  transactions[['Customer ID', 'Cohort Yr', 'Customer', 'Type', 'Date', 'Year',
                                'Month', 'Document Number', 'Description', 'Qty', 'Unit Price',
                                'Sales Total', 'Revenue', 'State']]


    transactions.to_csv('../data/processed/transactions.csv')
    print(transactions.head(),'\n transactions saved to ../data/processed/transactions.csv')


    credit_memos = transactions[transactions['Unit Price']<0]
    credit_memos.to_csv('../data/processed/credit_memos.csv')
    print(credit_memos.groupby('Year')['Qty', 'Sales Total'].sum().transpose())
    print('credit_memos saved to ../data/processed/credit_memos.csv \n')

    # Remove Credit Memos
    sales = transactions[transactions['Qty']>0]
    sales = sales[sales['Sales Total']>9]


    sales = sales.groupby(['Customer ID','Cohort Yr', 'Year','Document Number','Date','State']).sum('Sales Total').reset_index()
    print('Sales per year \n',sales.groupby('Year')['Sales Total'].sum(),'\n')

    AOV = sales.groupby('Year')['Sales Total'].sum() / sales.groupby('Year')['Document Number'].nunique() 
    print('AOV by year ',AOV,'\n')
    
    print('Sales mean by year \n',sales.groupby('Year')['Sales Total'].mean(),'\n')

    print('Avg order value 2020', sales[sales['Year']==2020]['Sales Total'].mean())
    AOV_2020 = sales[sales['Year']==2020]['Sales Total'].mean()

    sales['log_sales'] = np.log(sales['Sales Total'])


    print(sales.groupby('Cohort Yr')['log_sales'].mean())

    purchases = sales.groupby('Customer ID')['Document Number'].nunique()
    purchases.columns = ['Customer ID', 'purchases']
    sales = merge_dfs(sales,[purchases],'Customer ID')
    sales.rename({'Document Number_x': 'Document Number',
                'Document Number_y':'purchases'},
                axis=1,inplace=True)

    AOV_data = pd.DataFrame({'Annual Sales': sales.groupby('Year')['Sales Total'].sum(),
                                'AOV': AOV,
                                'Sales Total mean':sales.groupby('Year')['Sales Total'].mean(),
                                'log_sales avg':sales.groupby('Year')['log_sales'].mean(),
                                })

    print(AOV_data)

    fig, axs =plt.subplots(1,2, sharey=True, tight_layout=True)
    axs[0].hist(sales['Sales Total'])
    axs[0].set_title('Invoice Sales Total')


    axs[1].hist(sales['log_sales'])
    axs[1].set_title('Log of Invoice Sales Total')

    plt.show()

    sales.info()

    sales.to_csv('../data/processed/sales.csv')
    print('\n ',sales.head(),'\n saved to ../data/processed/sales.csv')

    
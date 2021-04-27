import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


# Transaction Analysis  df= transactions
def sales_per_year(df):
    annual_sales = pd.DataFrame(df.groupby("Year")['Sales Total'].sum()).reset_index()
    return annual_sales

def customer_serviced_yr(df): #use transactions
    customers_serviced = pd.DataFrame(df.groupby("Year")['Customer ID'].nunique()).reset_index()
    customers_serviced.columns = ['Year', 'Customer Count']
    return customers_serviced

def new_customers_yr(df):
    new_customers = pd.DataFrame(df.groupby('Cohort Yr')['Customer ID'].nunique()).reset_index()
    new_customers.columns = ['Year','New Customers']
    return new_customers

def new_customers_invoices(df):
    new_customers_orders = pd.DataFrame(df.groupby(['Year','Cohort Yr'])['Document Number'].nunique()).reset_index()
    new_customers_orders.columns = ['Year','Cohort Yr', 'Orders from new customers']
    new_customers_orders = new_customers_orders[new_customers_orders['Cohort Yr']== new_customers_orders['Year']]
    new_customers_orders = new_customers_orders[['Year', 'Orders from new customers']]
    return new_customers_orders

def new_customers_sales(df):
    sales_new_customers = pd.DataFrame(df.groupby(['Year','Cohort Yr'])['Sales Total'].sum()).reset_index()
    sales_new_customers.columns = ['Year', 'Cohort Yr','Sales from new customers']
    sales_new_customers = sales_new_customers[sales_new_customers['Cohort Yr'] == sales_new_customers['Year']]
    sales_new_customers = sales_new_customers[['Year', 'Sales from new customers']]
    return sales_new_customers



def orders_per_year(df):
    orders = pd.DataFrame(df.groupby("Year")['Document Number'].nunique()).reset_index()
    orders.columns = ['Year', 'Order Count']
    return orders

def merge_dfs(df,lst,key):
    for d in lst:
        df = df.merge(d, on=key)
    return df

def transaction_data(df):
    trans_data_table = merge_dfs(annual_sales, [customers_serviced,sales_new_customers,new_customers,orders, new_customers_orders], key='Year')
    trans_data_table['Year']  = trans_data_table['Year'].astype('int')
    trans_data_table['Percent of Sales from new Customers'] = (trans_data_table['Sales from new customers'] / trans_data_table['Sales Total'])*100
    trans_data_table['Avg Sales per Order'] = trans_data_table['Sales Total'] / trans_data_table['Order Count']
    trans_data_table['Avg num orders per Customer'] = trans_data_table['Order Count'] / trans_data_table['Customer Count']
    trans_data_table['Avg num orders per New Customer'] = trans_data_table['Orders from new customers'] / trans_data_table['New Customers']
    trans_data_table['Percent of order from New Customers'] = (trans_data_table['Orders from new customers'] / trans_data_table['Order Count'])*100
    trans_data_table['Average sales per new customer order'] = trans_data_table['Sales from new customers'] / trans_data_table['Orders from new customers']
    trans_data_table = trans_data_table.astype('int')
    return trans_data_table
#     transaction_data = pd.DataFrame()
#     Number of customers service per year      =   customers_serviced = customer_serviced_yr(transactions)
#     Number of new customers per year          =   cohort_by_customers = cohort_size(cohort_transactions)
#     Sales per year                            =   annual_sales = sales_per_year(transactions)
#     Orders per year                           =   orders = orders_per_year(transactions)
#     Average sales per order                   =   annual_sales / orders
#     Average # orders per customer             =   customers_serviced / 
#     Average sales per new customer order      
#     Average number of orders per new customer 



# Visuals

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



if __name__ == "__main__":  # used for testing purposes but will transition to OOP

    df = pd.read_csv('../data/processed/sales.csv')
    print(df.head(),'\n' )

    annual_sales = sales_per_year(df)
    customers_serviced = customer_serviced_yr(df)
    orders = orders_per_year(df)
    new_customers = new_customers_yr(df)
    new_customers_orders = new_customers_invoices(df)
    print('\n',new_customers_orders.transpose())
    sales_new_customers = new_customers_sales(df)
    print('\n',sales_new_customers.transpose())

    print('\n\n')



    trans_data_table = transaction_data(df)
    trans_data_table.transpose().to_csv('../data/processed/transaction_analysis_pivot.csv')
    print(trans_data_table.transpose())
    

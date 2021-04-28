import numpy as np
import pandas as pd
import datetime as dt 
import matplotlib.pyplot as plt
import os
import lifetimes
import lifetimes.utils
import lifetimes.fitters
import lifetimes.plotting
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from cohort_analysis import customer_cohort_data
from cohort_analysis import customer_cohort_chart
from cleaning import merge_dfs
from math import e 

def load_data(path):
    df = pd.read_csv(path)
    print(df.head(),'\n',df.shape,'\n data loaded')
    return df 

def grab_cohorts(df,column_name):
    cohort = pd.DataFrame(df).groupby('Customer ID')[column_name].min().reset_index()
    return cohort 

# def rfm_process_python(df):
#     rfm = df.groupby('Customer ID').agg({'Date': lambda x: (x.max() - x.min()).days,
#                                         'Document Number': lambda x: len(x),
#                                         'Sales Total': lambda x: sum(x)}
#                                         )
#     rfm.columns = ['recency', 'frequency', 'monetary_value']
#     print(rfm) 
#     return rfm 

def rfm_process(df):
    #https://lifetimes.readthedocs.io/en/latest/lifetimes.html#module-lifetimes.utils
    import lifetimes 
    rfm = lifetimes.utils.summary_data_from_transaction_data(df,
                                                            customer_id_col='Customer ID', 
                                                            datetime_col='Date',
                                                            monetary_value_col= 'Sales Total',  #repeated_transactions.groupby(customer_id_col)[monetary_value_col].mean().fillna(0)
                                                            observation_period_end=None,
                                                            freq='D',
                                                            freq_multiplier=1
                                                            )
    mon_val_of_zero = round(rfm[rfm['monetary_value']==0]['monetary_value'].count(),2)
    percent_of_zero_rfm = round((mon_val_of_zero / len(rfm)) * 100,2)
    print(rfm.head(),'\n')

    if percent_of_zero_rfm > .05: 
        rfm = rfm[rfm['monetary_value']>0]
    rfm
    print(rfm.head(),'\n')
    print(len(rfm),' repeat customers')
    print('\nmean\n',rfm.describe())

    return rfm

def mean_median_std(df):
    df_stats = pd.DataFrame({'mean':round(rfm['monetary_value'].mean(),2),
                            'median':round(rfm['monetary_value'].median(),2),
                            'std':round(rfm['monetary_value'].std(),2),
                            'min':round(rfm['monetary_value'].min(),2),
                            'max':round(rfm['monetary_value'].max(),2)},
                            index=['monetary_value'])
    print(df_stats)
    return df_stats

def split(df):
    # https://lifetimes.readthedocs.io/en/latest/lifetimes.html#module-lifetimes.utils
    '''
    calibration_and_holdout_data() function returns df with following _cal as calibration columns and _holdout as holdout columns:
      frequency_cal
      recency_cal
      T_cal
      frequency_holdout  
      duration_holdout = End of Period - First purchase date
    '''
    train_test = lifetimes.utils.calibration_and_holdout_data(df,
                                                        'Customer ID',
                                                        'Date', 
                                                        calibration_period_end= training_end,
                                                        observation_period_end=validation_end,
                                                        monetary_value_col='Sales Total')
    train_test = train_test[train_test['frequency_cal']>0]
    print(train_test.head())
    print(train_test.shape)
    return train_test

def thresh_prediction(df,thresh):
    alive_prediction = pd.DataFrame(df[df['prob_alive']>thresh].groupby('Cohort Yr')['prob_alive'].sum())
    purchase_prediction = pd.DataFrame(df[df['prob_alive']>thresh].groupby('Cohort Yr')['pred_purchases'].mean())
    alive_purchasing_prediction =  pd.concat([alive_prediction,purchase_prediction],axis=1)
    alive_purchasing_prediction.columns=['predicted customer count', 'predicted avg. number of purchases']
    return alive_purchasing_prediction


if __name__ == "__main__":
    # Assumptions/Drivers
    training_end ='2018-12-31'
    validation_end ='2020-12-31'
    WACC = .25
    monthly_discount_rate = WACC/12
    profit_margin = 0.26
    t_params= [ 1, 30, 90, 365 ]


    
    # LOAD AND PROCESS DATA
    path = '../data/processed/sales.csv'
    df = load_data(path)
    
    cohort = grab_cohorts(df,'Year')
    cohort.columns =['Customer ID', 'Cohort Yr']
    Cohort_yr = cohort 

    rfm = rfm_process(df)
    rfm = pd.merge(rfm,Cohort_yr, on='Customer ID')
    # rfm.to_csv('../data/processed/rfm.csv')
    print('Data Processed')

    rfm_stats = mean_median_std(rfm)

    print('  SPLIT  ')
    rfm_train_test = split(df)




    print('\n\n BGF Model fit ') # frequency, recency
    # ---------------------------------------------------------------------------------------------------------------------------
    ## baseline model
    bgf = lifetimes.BetaGeoFitter(penalizer_coef=0.0) # <-regression hyperparameter
    # https://lifetimes.readthedocs.io/en/latest/lifetimes.fitters.html#module-lifetimes.fitters.beta_geo_fitter
    bgf.fit(rfm_train_test['frequency_cal'], rfm_train_test['recency_cal'], rfm_train_test['T_cal'])
    # bgf.fit(train) .fit doesn't accept variable TypeError: fit() missing 2 required positional arguments: 'recency' and 'T'
    print(bgf.summary,'\n')
    



    print('\n\n BGF predict frequency') 
    t =365 #Calculate the expected number of repeat purchases up to time t
    predicted_bgf = round(bgf.conditional_expected_number_of_purchases_up_to_time(t, 
                                                                              rfm_train_test['frequency_cal'],
                                                                              rfm_train_test['recency_cal'],
                                                                              rfm_train_test['T_cal']))
    actual_freq = rfm_train_test['frequency_holdout']
    predicted_freq = predicted_bgf
    baseline_penalizer_score = metrics.mean_absolute_error(actual_freq, predicted_freq)
    print('MAE = ', round(baseline_penalizer_score,5))
    # ---------------------------------------------------------------------------------------------------------------------------






    print('\n\n Evaluate / tune hyperparameters for frequency ' )
    # ---------------------------------------------------------------------------------------------------------------------------
    params = [0.0, 0.001, 0.1]
    list_of_f = []
    case = 0
    for p in params:
        bgf = lifetimes.BetaGeoFitter(penalizer_coef=p)
        bgf.fit(rfm_train_test['frequency_cal'], rfm_train_test['recency_cal'], rfm_train_test['T_cal'])
        predicted_bgf = round(bgf.conditional_expected_number_of_purchases_up_to_time(t,
                                                                              rfm_train_test['frequency_cal'],
                                                                              rfm_train_test['recency_cal'],
                                                                              rfm_train_test['T_cal']))
        predicted_freq = predicted_bgf
        f_df = pd.DataFrame({'p'                          : p,
                            '2020_Actual_Avg_Frequency'   : round(actual_freq.mean(),4),
                            '2020_Predicted_Avg_Frequency': round(predicted_freq.mean(),4),
                            'Average absolute error'      : metrics.mean_absolute_error(actual_freq, predicted_freq)
                          },index=[p])
        list_of_f.append(f_df)
        f_penalizer_df = pd.concat(list_of_f)
    print(f_penalizer_df.transpose())
    f_best_penalizer_score = f_penalizer_df['Average absolute error' ].min()

    f_best_penalizer = f_penalizer_df['Average absolute error' ].idxmin()
    
    print('\nbest penalizer = ', f_best_penalizer)
    print('best score ',f_best_penalizer_score)

    rfm_train_test['predicted purchases test'] = predicted_bgf
    print('Predicted purchases for test period ', rfm_train_test['predicted purchases test'].sum())



    print('\n\n Fit frequency Best Model ')
    # ---------------------------------------------------------------------------------------------------------------------------
    bgf = lifetimes.BetaGeoFitter(penalizer_coef=f_best_penalizer) # <-regression hyperparameter
    bgf.fit(rfm_train_test['frequency_cal'], rfm_train_test['recency_cal'], rfm_train_test['T_cal'])
    # bgf.fit(train) .fit doesn't accept variable TypeError: fit() missing 2 required positional arguments: 'recency' and 'T'
    print(bgf.summary,'\n')

    lifetimes.plotting.plot_calibration_purchases_vs_holdout_purchases(bgf,rfm_train_test)
    plt.show()



    print('\n\n Predict frequency & recency with Best Model') 
    # ------------------------------------------------------------------------------------------------------------------------------

    t =365
    rfm_train_test['predicted_purchases'] = round(bgf.conditional_expected_number_of_purchases_up_to_time(
                                                                              t,
                                                                              rfm_train_test['frequency_cal'],
                                                                              rfm_train_test['recency_cal'],
                                                                              rfm_train_test['T_cal']))


    rfm_train_test['probability_alive'] = bgf.conditional_probability_alive(rfm_train_test['frequency_cal'],
                                                                              rfm_train_test['recency_cal'],
                                                                              rfm_train_test['T_cal'])

#     print(rfm_actuals.sort_values(by='predicted_purchases').tail())

    print('Hold Out period Predicted Mean probability a customer is alive is ' , round(rfm_train_test['probability_alive'].mean(),2)*100,'%')
    print('Hold Out period Predicted Mean predicted purchases' , round(rfm_train_test['predicted_purchases'].mean(),2))
    # ------------------------------------------------------------------------------------------------------------------------------




    print('\n\n  GAMMA-GAMMA MODEL')


    # Fit Gamma-Gamma Model for Monetary Value
    # ------------------------------------------------------------------------------------------------------------------------------
    #confirm no correlation
    print(rfm_train_test[['monetary_value_cal', 'frequency_cal']].corr())
    # insert unit test
    rfm_train_test = rfm_train_test[rfm_train_test['monetary_value_cal'] >=1] #filter out negatives


    ggf = lifetimes.fitters.gamma_gamma_fitter.GammaGammaFitter(penalizer_coef = 0.0001)
    ggf.fit(rfm_train_test['frequency_cal'],
            rfm_train_test['monetary_value_cal'])
    
    print('\n')
    print(ggf.params_)
    print(len(ggf.data),' rows')
    # ------------------------------------------------------------------------------------------------------------------------------




    # PREDICT Gamma-Gamma Model for Monetary Value
    # ------------------------------------------------------------------------------------------------------------------------------
    monetary_pred = ggf.conditional_expected_average_profit(rfm_train_test['frequency_holdout'],
                                                       rfm_train_test['monetary_value_holdout'])


    test_m = rfm_train_test['monetary_value_holdout']

    # Evaluate & tune parameters for Gamma-Gamma Model prediction
    # ------------------------------------------------------------------------------------------------------------------------------   
    params = [0.0, 0.0001, 0.001, 0.005, 0.01,  0.5, 1]
    list_of_m = []
    case = 0
    for p in params:
        ggf = lifetimes.fitters.gamma_gamma_fitter.GammaGammaFitter(penalizer_coef=p)
        ggf.fit(rfm_train_test['frequency_cal'],
                rfm_train_test['monetary_value_cal'])


        predicted_m  = ggf.conditional_expected_average_profit(rfm_train_test['frequency_holdout'],
                                                       rfm_train_test['monetary_value_holdout'])
        
        

        case +=1
        m_df = pd.DataFrame({'p'                               : p,
                            # '2020_Actual_Monetary_Value'   : round(rfm_train_test['monetary_value_holdout'].mean(),4),
                            # '2020_Predicted_Monetary_Value': round(predicted_m,4),
                            'Average absolute error'           : metrics.mean_absolute_error(rfm_train_test['monetary_value_holdout'], predicted_m)
                          },index=[p])

        list_of_m.append(m_df)
        m_penalizer_df = pd.concat(list_of_m)
    print(m_penalizer_df.transpose())
    m_best_penalizer_score = m_penalizer_df['Average absolute error' ].min()

    m_best_penalizer = m_penalizer_df['Average absolute error' ].idxmin()
    
    print('\nbest penalizer = ', m_best_penalizer)
    print('best score ',m_best_penalizer_score)








    # Fit gamma-gamma best model to train_test
    # -------------------------------------------------------------------------------------------------------------------------------
    ggf = lifetimes.fitters.gamma_gamma_fitter.GammaGammaFitter(penalizer_coef = m_best_penalizer)
    ggf.fit(rfm_train_test['frequency_cal'],
            rfm_train_test['monetary_value_cal'])

    best_monetary_pred = ggf.conditional_expected_average_profit(rfm_train_test['frequency_holdout'],
                                                        rfm_train_test['monetary_value_holdout'])

    rfm_train_test['predicted avg order value'] = best_monetary_pred                                                     
    # -------------------------------------------------------------------------------------------------------------------------------
    

    # Evaluate best models against holdout year
    # -------------------------------------------------------------------------------------------------------------------------------

    rfm_train_test['predicted test revenue'] = (rfm_train_test['predicted avg order value'] * rfm_train_test['predicted purchases test']) / rfm_train_test['probability_alive']
    
    print('Predicted avg order value for test period ', rfm_train_test['predicted avg order value'].mean())
    print('Predicted purchases for test period ', rfm_train_test['predicted purchases test'].sum())
    print('\nPredicted test revenue', rfm_train_test['predicted test revenue'].sum())

    # ------------------------------------------------------------------------------------------------------------------------------   



    # Fit monetary_value Best Model
    # ------------------------------------------------------------------------------------------------------------------------------   
    rfm = rfm.loc[rfm['monetary_value'] > 0]

    ggf = lifetimes.fitters.gamma_gamma_fitter.GammaGammaFitter(penalizer_coef = m_best_penalizer)
    ggf.fit(rfm['frequency'],
            rfm['monetary_value'])
    

    print('\n')
    print(ggf.params_)
    print(len(ggf.data),' rows')
    # ------------------------------------------------------------------------------------------------------------------------------   


    print(rfm[rfm['monetary_value']<=0])

    # Predict M 
    # ------------------------------------------------------------------------------------------------------------------------------   
    returning_customers = rfm[rfm['frequency']>0]

    print('Number of returning customers ',len(returning_customers))
    
    ggf = lifetimes.fitters.gamma_gamma_fitter.GammaGammaFitter(penalizer_coef=m_best_penalizer)
    ggf.fit(rfm['frequency'],rfm['monetary_value'])
    
    returning_customers['predicted_m'] = ggf.conditional_expected_average_profit(returning_customers['frequency'],
                                                                                returning_customers['monetary_value'])




    # CLV MODEL

    # Train CLTV
    # ------------------------------------------------------------------------------------------------------------------------------   
    clv = ggf.customer_lifetime_value(
                                    bgf,
                                    rfm_train_test['frequency_cal'],
                                    rfm_train_test['recency_cal'],
                                    rfm_train_test['T_cal'],
                                    rfm_train_test['monetary_value_cal'],
                                    time=1200, # time in months - see issue #3 http://www.brucehardie.com/notes/033/what_is_wrong_with_this_CLV_formula.pdf 
                                    discount_rate=0.01
                                    )
    rfm_train_test['lifetime Sales'] = clv

    print(rfm_train_test.head())

    print('\n')

    print('lifetime Sales ',round(rfm_train_test['lifetime Sales'].sum()))
    # need to evalauate CLV





    # Retraining the Model
    # ------------------------------------------------------------------------------------------------------------------------------   
    #RFM
    rfm = lifetimes.utils.summary_data_from_transaction_data(df, 
                                                                customer_id_col = 'Customer ID',
                                                                datetime_col = 'Date',
                                                                monetary_value_col= 'Sales Total',
                                                                observation_period_end=None,
                                                                freq='D',
                                                                freq_multiplier=1)
    
    #print('MONETARY VALUE < 0 \n ',rfm[rfm['monetary_value']<0])
    rfm = rfm.loc[rfm.monetary_value > 0, :] # model is sensitive to negative monetary values


    #BG/NBD
    bgf = lifetimes.BetaGeoFitter(penalizer_coef=f_best_penalizer)
    bgf.fit(rfm['frequency'], rfm['recency'], rfm['T'])
    rfm['prob_alive'] = bgf.conditional_probability_alive(frequency=rfm['frequency'],recency=rfm['recency'],T=rfm['T'])
    rfm['pred_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t=365,
                                                                                 frequency=rfm['frequency'],
                                                                                 recency=rfm['recency'],
                                                                                 T=rfm['T'])


    print(rfm.sort_values(by='frequency').tail())
    from lifetimes.plotting import plot_history_alive

    id = 100568
    days_since_birth = 2178
    sp_trans = df.loc[df['Customer ID'] == id]
    df["Date"] = df['Date'].astype(str)

    # alive_plot = plot_history_alive(bgf, days_since_birth, sp_trans, 'Date', c='b')






    #GG
    ggf = lifetimes.GammaGammaFitter(penalizer_coef=m_best_penalizer)
    ggf.fit(rfm['frequency'], rfm['monetary_value'])


    #CLV Model
    rfm['exp_avg_order_value'] = ggf.conditional_expected_average_profit(rfm['frequency'],rfm['monetary_value'])
    clv = ggf.customer_lifetime_value(
                                    bgf,
                                    rfm['frequency'],
                                    rfm['recency'],
                                    rfm['T'],
                                    rfm['monetary_value'],
                                    time=1200,  # the lifetime expected for the user in months. Default: 12
                                    freq='D',
                                    discount_rate=0.01
                                    )

    rfm['Predicted next year sales'] = (rfm['pred_purchases'] * rfm['exp_avg_order_value'])#/( 1- rfm['prob_alive'] )
    rfm['CLT Sales'] = clv
    rfm['CLTV'] = clv * profit_margin

    #pd.set_option('display.float_format', lambda x: '%.0f' % x)

    rfm = rfm.merge(Cohort_yr, on="Customer ID",how='inner')
#     print(rfm.head())
#     print(len(rfm))




    print('\n', round(rfm.describe(),2))
    rfm.to_csv('../data/processed/rfm.csv')
    print('rfm.csv saved \n')

    print('Predicted Purchases ',round(rfm['pred_purchases'].sum(),2))
    print('Predicted next year sales ',round(rfm['Predicted next year sales'].sum(),2))
    cohort_prediction = rfm.groupby('Cohort Yr')['Predicted next year sales'].sum().reset_index()
    cohort_prediction.columns = ['Cohort Yr', 'P2021']
    cohort_prediction['Percent of Next Year Sales'] = (cohort_prediction['P2021'] / rfm['Predicted next year sales'].sum())*100
    cohort_prediction.to_csv('../data/processed/cohort_predictiosn.csv')
    print(round(cohort_prediction))
#     print('CLT Sales 12 months' , rfm['CLT Sales'].sum())

#     print(round(rfm[rfm['prob_alive']>0.6].count()))

    print('CLTV ',round(rfm['CLTV'].sum(),2))


    cust_count_pred = rfm.groupby('Cohort Yr')['prob_alive'].value_counts(bins=3)
    cust_count_pred =  cust_count_pred.reset_index()
    cust_count_pred.columns = ['Cohort Yr','Probability Alive', 'Customer Count']

    # cust_prediction_purchases = rfm.groupby('Cohort Yr')['pred_purchases'].value_counts(bins=3)
    # cust_prediction_purchases = cust_prediction_purchases.reset_index()
    # cust_prediction_purchases.columns = ['Cohort Yr','Probability of Purchase', 'Number of Purchases']

    # prediction_cohort_analysis = cust_count_pred.merge(cust_prediction_purchases, on='Cohort Yr')
    # print(prediction_cohort_analysis)


    # alive_purchasing_prediction = thresh_prediction(rfm,thresh=0.6)
    # print(alive_purchasing_prediction)
    # alive_purchasing_prediction.to_csv('../data/processed/alive_purchasing_prediction.csv')




    # Predict 

    # predicted purchases
    # probability alive
    # clv at time t less CaC
    



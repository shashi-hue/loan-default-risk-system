import pandas as pd
import numpy as np


def preprocess_pipeline(input_path,usecols=None):

    df = pd.read_csv(input_path,low_memory=False,usecols=usecols)


    df['credit_card_util_pct'] = df['revol_bal'] / (df['total_rev_hi_lim'] + 1)
    df['installment_income_ratio'] = df['installment'] / (df['annual_inc'] + 1)
    df['loan_amount_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)



    df['had_recent_major_derog'] = (df['mths_since_last_major_derog'] < 12).astype(int)
    df['delinq_flag'] = (df['mths_since_last_delinq'] < 12).astype(int)
    df['bankruptcy_flag'] = (df['pub_rec_bankruptcies'] > 0).astype(int)
    df['historical_delinquency_rate'] = df['delinq_2yrs'] / (df['total_acc'] + 1)



    df['recent_open_accs'] = (
        df[['open_acc_6m', 'open_il_12m', 'open_il_24m', 'open_rv_12m', 'open_rv_24m']]
        .sum(axis=1, skipna=True)
    )

    df['recent_inquiries_sum'] = (
        df[['inq_last_6mths', 'inq_fi', 'inq_last_12m', 'sec_app_inq_last_6mths']]
        .sum(axis=1, skipna=True)
    )



    df['is_major_derog_and_util_high'] = (
        ((df['mths_since_last_major_derog'] < 12) & (df['revol_util'] > 80))
    ).astype(int)

    df['desc_contains_debt'] = df['desc'].fillna('').str.contains(r'debt', case=False).astype(int)

    ### Cleaning

    keep_statuses = [
        'Fully Paid', 'Charged Off', 'Default',
        'Does not meet the credit policy. Status: Fully Paid',
        'Does not meet the credit policy. Status: Charged Off'
    ]

    df = df[df['loan_status'].isin(keep_statuses)]

    default_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status: Charged Off']

    df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x in default_statuses else 0)

    df['home_ownership'] = df['home_ownership'].replace({'ANY': 'OTHER', 'NONE': 'OTHER', 'OTHER': 'OTHER'})

    ### Encoding and transformations

    one_hot_cols = ['verification_status','home_ownership','purpose']
    df = pd.get_dummies(df,columns=one_hot_cols,drop_first=True)

    emp_map = {
        '< 1 year': 0,
        '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8,
        '9 years': 9, '10+ years': 10
    }
    df['emp_length'] = df['emp_length'].map(emp_map)

    df['application_type'] = df['application_type'].map({'Individual' : 0,'Joint App':1})
    df['term'] = df['term'].map({' 36 months' : 0, ' 60 months' : 1})

    df = df[df['dti']>0]

    addr_freq = df['addr_state'].value_counts(normalize=True)
    df['addr_state_freq'] = df['addr_state'].map(addr_freq)
    df = df.drop(columns=['addr_state'])

    subgrade_order = []
    for g_idx, g in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 1):
        for s in range(1, 6):
            subgrade_order.append(f"{g}{s}")

    subgrade_float_map = {sg: float(f"{g_idx}.{s}") for sg, (g_idx, s) in zip(subgrade_order, [(i//5 + 1, i%5 + 1) for i in range(35)])}

    df['sub_grade'] = df['sub_grade'].map(subgrade_float_map)



    df = df.drop(columns=['grade'])


    df['revol_util'] = df['revol_util'].clip(upper=100)



    df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    df.drop(['fico_range_low', 'fico_range_high'], axis=1, inplace=True)


    df['fico_dti_ratio'] = df['fico_score'] / (1 + df['dti'])



    df['annual_inc_log'] = np.log1p(df['annual_inc'])
    df = df.drop(columns=['annual_inc'])

    df['dti_log'] = np.log1p(df['dti'])
    df = df.drop(columns=['dti'])





    df['collections_12_mths_ex_med'] = df['collections_12_mths_ex_med'].fillna(0)

    df['collections_bins'] = pd.cut(df['collections_12_mths_ex_med'],
                                        bins=[-1,0,1,np.inf],
                                        labels=[0,1,2]).astype(int)
    df.drop(columns=['collections_12_mths_ex_med'], inplace=True)


    df['revol_bal_log'] = np.log1p(df['revol_bal'])
    df = df.drop(columns=['revol_bal'])


    df['tot_coll_amt_log'] = np.log1p(df['tot_coll_amt'])
    df = df.drop(columns=['tot_coll_amt'])


    df['tot_cur_bal_log'] = np.log1p(df['tot_cur_bal'])
    df = df.drop(columns=['tot_cur_bal'])


    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'],format='%b-%Y', errors='coerce')
    df['issue_d'] = pd.to_datetime(df['issue_d'],format='%b-%Y', errors='coerce')

    df['account_age_days'] = (df['issue_d'] - df['earliest_cr_line']).dt.days
    df['account_age_years'] = df['account_age_days'] / 365.25


    df['sec_app_earliest_cr_line'] = pd.to_datetime(df['sec_app_earliest_cr_line'], format='%b-%Y', errors='coerce')


    df['sec_app_credit_age_days'] = (df['issue_d'] - df['sec_app_earliest_cr_line']).dt.days
    df['sec_app_credit_age_years'] = df['sec_app_credit_age_days'] / 365.25


    df.drop(['earliest_cr_line', 'issue_d','account_age_days','sec_app_earliest_cr_line','sec_app_credit_age_days'], axis=1, inplace=True)

    df['annual_inc_joint_log'] = np.log1p(df['annual_inc_joint'])
    df['dti_joint_log'] = np.log1p(df['dti_joint'])

    df = df.drop(columns=['annual_inc_joint','dti_joint'])

    df = pd.get_dummies(df,columns=['verification_status_joint'],dummy_na=True)

    df['revol_bal_joint_log'] = np.log1p(df['revol_bal_joint'])
    df = df.drop(columns=['revol_bal_joint'])

    df['sec_app_fico_score'] = (df['sec_app_fico_range_low'] + df['sec_app_fico_range_high']) / 2
    df.drop(['sec_app_fico_range_low', 'sec_app_fico_range_high'], axis=1, inplace=True)

    df['sec_app_revol_util'] = df['sec_app_revol_util'].clip(upper=100)

    df['sec_app_collections_bins'] = pd.cut(
        df['sec_app_collections_12_mths_ex_med'],
        bins=[-1, 0, 1, np.inf],
        labels=[0, 1, 2]
    )

    # convert labels to integers, preserving NaN
    df['sec_app_collections_bins'] = df['sec_app_collections_bins'].astype('float')

    df.drop(columns=['sec_app_collections_12_mths_ex_med'], inplace=True)


    emp_title_counts = df['emp_title'].value_counts()
    df['emp_title_freq'] = df['emp_title'].map(emp_title_counts)

    df.drop(columns=['emp_title'], inplace=True)

    df['has_desc'] = df['desc'].notnull().astype(int)




    df.drop(columns=['desc'], inplace=True)

    ### More features


    df['inq_last_6mths_missing'] = df['inq_last_6mths'].isna().astype(int)
    df['mths_since_last_delinq_missing'] = df['mths_since_last_delinq'].isna().astype(int)
    df['mths_since_last_record_missing'] = df['mths_since_last_record'].isna().astype(int)
    df['revol_util_missing'] = df['revol_util'].isna().astype(int)
    df['mths_since_last_major_derog_missing'] = df['mths_since_last_major_derog'].isna().astype(int)
    df['num_tl_90g_dpd_24m_missing'] = df['num_tl_90g_dpd_24m'].isna().astype(int)


    df['revolutil_fico'] = df['revol_util'] * df['fico_score']

    df['dti_loaninc_ratio'] = df['dti_log'] * df['loan_amount_income_ratio']

    df['delinq_coll_bin'] = df['delinq_flag'] * df['collections_bins']

    df['emplen_verif'] = df['emp_length'] * df['verification_status_Verified']

    df['inst_inc_purpose_debtcon'] = df['installment_income_ratio'] * df['purpose_debt_consolidation']

    df['accage_openacc'] = df['account_age_years'] * df['open_acc']

    return df


def run_preprocessing():
    model_features = [
    # Borrower info
    'annual_inc', 'verification_status', 'emp_length', 'home_ownership', 'addr_state',
    'dti', 'purpose', 'application_type', 'emp_title', 'desc',

    # Credit risk indicators
    'fico_range_low', 'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
    'mths_since_last_record', 'open_acc', 'total_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'collections_12_mths_ex_med', 'mths_since_last_major_derog',
    'delinq_2yrs', 'acc_now_delinq', 'num_tl_90g_dpd_24m', 'policy_code',
    'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m',
    'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m',
    'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi',
    'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal',
    'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt',
    'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl', 'earliest_cr_line', 'mort_acc', 'mths_since_recent_bc',
    'mths_since_recent_bc_dlq', 'mths_since_recent_inq',
    'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 'num_actv_bc_tl',
    'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
    'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',
    'num_tl_30dpd', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq',
    'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim',
    'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit','issue_d',

    # Loan application terms
    'loan_amnt', 'int_rate', 'term', 'installment', 'grade', 'sub_grade',

    # joint and secondary applicant info
    'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'revol_bal_joint',
    'sec_app_fico_range_low', 'sec_app_fico_range_high', 'sec_app_earliest_cr_line',
    'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
    'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts',
    'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med',
    'sec_app_mths_since_last_major_derog',

    # status
    'loan_status'
    ]

    input_path = "data/raw/accepted_2007_to_2018Q4.csv"
    output_path = "data/processed/model_df.parquet"

    df = preprocess_pipeline(input_path,usecols=model_features)

    df.to_parquet(output_path)

    print(f"✅ Preprocessing complete. Saved to {output_path}")

if __name__ == "__main__":
    run_preprocessing()
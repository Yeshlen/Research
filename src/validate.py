"""Basic dataset checks used by notebook 01; no fitted classifier or statistical diagnostic."""
import pandas as pd


def constraint_violations(df, cfg):
    p = cfg['product']
    red = df.redeemed == 1
    dates = pd.to_datetime(df.created_at)
    days = df.assign(calendar_day=dates.dt.date).groupby(['customer_id', 'calendar_day']).amount.sum()
    months = df.assign(calendar_month=dates.dt.to_period('M')).groupby(['customer_id', 'calendar_month']).amount.sum()
    return {
        'amount not a multiple of step': int((df.amount % p['amount_step'] != 0).sum()),
        'amount not positive': int((df.amount <= 0).sum()),
        'amount above max daily limit': int((df.amount > p['max_daily_limit']).sum()),
        'amount above current limit': int((df.amount_to_limit_ratio > 1.0001).sum()),
        'calendar-day total above max limit': int((days > p['max_daily_limit']).sum()),
        'calendar-month total above limit': int((months > p['monthly_limit']).sum()),
        'redeemed but no delay': int((red & df.redeem_delay_min.isna()).sum()),
        'not redeemed but has delay': int((~red & df.redeem_delay_min.notna()).sum()),
        'negative redemption delay': int((df.redeem_delay_min < 0).sum()),
        'redeemed after expiry': int((df.redeem_delay_min > p['voucher_expiry_days'] * 1440).sum()),
        'recipient known with no history': int(((df.recipient_known == 1) & (df.history_count == 0)).sum()),
        'self-send flagged as added recipient': int(((df.is_self_send == 1) & (df.recipient_added_in_session == 1)).sum()),
        'voucher before account opened': int((df.account_age_days < 0).sum()),
        'missing label': int(df.is_scam.isna().sum()),
        'invalid label': int((~df.is_scam.isin([0, 1])).sum()),
        'duplicate voucher features': int(df.voucher_id.duplicated().sum()),
    }


def check_dataset(df, truth, cfg):
    checks = constraint_violations(df, cfg)
    checks['duplicate truth vouchers'] = int(truth.voucher_id.duplicated().sum())
    checks['voucher IDs absent from truth'] = len(set(df.voucher_id) - set(truth.voucher_id))
    checks['truth vouchers absent from features'] = len(set(truth.voucher_id) - set(df.voucher_id))
    joined = df[['voucher_id', 'is_scam']].merge(
        truth[['voucher_id', 'is_scam']].drop_duplicates('voucher_id'), on='voucher_id', suffixes=('_feature', '_truth'))
    checks['labels different from truth'] = int((joined.is_scam_feature != joined.is_scam_truth).sum())
    return pd.DataFrame({'check': list(checks), 'violations': list(checks.values())})

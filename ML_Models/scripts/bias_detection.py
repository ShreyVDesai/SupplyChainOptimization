import pandas as pd
import send_email from utils
def handle_bias(rmses: pd.Series):
    std = rmses.std()
    if std>=5:
        send_email(emailid='talksick530@gmail.com',body='Bias detected')
    else:
        send_email(emailid='talksick530@gmail.com',body='Bias not detected')
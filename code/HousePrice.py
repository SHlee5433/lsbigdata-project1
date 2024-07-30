import pandas as pd
import numpy as np

house_df = pd.read_csv("./data/houseprice/train.csv")
<<<<<<< HEAD
=======
house_df.shape
>>>>>>> 538e365b24b2cd5f29d69b3d977f3f6d25976a53

price_mean = house_df["SalePrice"].mean()
price_mean

sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

sub_df["SalePrice"] = price_mean
sub_df

sub_df.to_csv("./data/houseprice/sample_submission.csv", index = False)
sub_df
<<<<<<< HEAD

=======
>>>>>>> 538e365b24b2cd5f29d69b3d977f3f6d25976a53

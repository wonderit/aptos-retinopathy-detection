import hashlib
from os.path import join

import pandas as pd
import psutil
from joblib import Parallel, delayed

# path to original APTOS 2019 train label file
train_df = pd.read_csv("../input/trainLabels19.csv")

# path to folder that contains original (unedited) APTOS 2019 images
train_path = "/path/to/2019_train/images/folder"


# train_df["diagnosis"].value_counts()


def get_hash(file):
    with open(file, "rb") as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
    return md5_hash


def get_full_path(path, file_name):
    return join(path, file_name + ".png")


train_file_hashes = Parallel(n_jobs=psutil.cpu_count(), verbose=1)(
    (delayed(get_hash)(get_full_path(train_path, x)) for x in train_df.id_code))

train_df["md5"] = train_file_hashes
train_df['md5_count'] = train_df.groupby('md5').id_code.transform('count')
train_df['md5_dub_no_unique'] = train_df.groupby('md5').diagnosis.transform('nunique').astype('int')

df_uni = train_df[(train_df.md5_count > 1) & (train_df.md5_dub_no_unique > 1)]
len(df_uni), df_uni['md5'].nunique()

df_train_no_dub = pd.DataFrame(train_df.drop_duplicates(subset=['md5', 'diagnosis'], keep='first'))
df_train_no_dub['md5_count'] = df_train_no_dub.groupby('md5').id_code.transform('count')

df_train_no_dub.reset_index(inplace=True)
df_train_no_dub.drop(columns="index", inplace=True)

df_train_no_dub[df_train_no_dub.md5_count > 1].md5_count.value_counts()

df_train_dublicates = df_train_no_dub[df_train_no_dub.md5_count > 1].sort_values("md5")

df_train_dublicates.to_csv("../input/trainLabels19_duplicates.csv",
                           columns=["id_code", "diagnosis", "md5"],
                           index=False)

df_train_unique = pd.DataFrame(df_train_no_dub.drop_duplicates(subset=['md5'], keep=False))

df_train_unique.to_csv("../input/trainLabels19_unique.csv",
                       columns=["id_code", "diagnosis"], index=False)

import os
import pandas as pd
dict1 = {'loss': 2.2195766342263066,
'acc1' : 39.91935483870968,
'acc5' : 75.80645161290323}
print(dict1)
path_=r'D:\Finarb\Azure_name_classification_V1\ndc-det-v0.2.1\outputs\CE'
df=pd.DataFrame.from_dict(dict1)
df.to_csv(os.path.join(path_, "class_metrics.csv"), index=False, header=True)
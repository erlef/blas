import pandas as pd

db = pd.read_csv("../lapacke_backend/total.csv", index_col=0)
db = db[~ db['arg_type'].map(lambda x: ('LAPACK' in x))]
db = db[db['fct_name'].map(lambda x: 'LAPACK' in x)]

fcts_unique_args = db.groupby('fct_name')['arg_type'].unique().map(lambda l: set(l)).reset_index()
types_to_test = set(db['arg_type'].unique())
fct_to_test = []

while len(types_to_test) > 0:
    print(types_to_test)
    # select the function with the highest number of different arguments
    n_diff_args = fcts_unique_args['arg_type'].map(lambda k: len(k))
    n_diff_args = n_diff_args[n_diff_args < 4]
    selected_id = n_diff_args.idxmax()
    fct_to_test.append(fcts_unique_args.iloc[selected_id]['fct_name'])

    selected = fcts_unique_args.iloc[selected_id]['arg_type']
    fcts_unique_args['arg_type'] = fcts_unique_args['arg_type'].apply(lambda x: x - selected)
    types_to_test = types_to_test - selected

print(fct_to_test)
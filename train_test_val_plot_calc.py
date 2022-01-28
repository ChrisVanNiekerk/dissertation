import pandas as pd

# %%
def actual_tree_data(sigma):

    tree_actual_df_path = 'data/EG0181T Riverdale A9b MAIN.xlsx'
    tree_actual_df = pd.read_excel(open(tree_actual_df_path, 'rb'), sheet_name='Data Rods versus Drone')
    last_valid_entry = tree_actual_df['Plot'].last_valid_index()
    tree_actual_df = tree_actual_df.loc[0:last_valid_entry]
    tree_actual_df = tree_actual_df.astype({'Plot':'int','Rep':'int','Tree no':'int'})
    tree_actual_df['tree_id'] = tree_actual_df['Plot'].astype('str') + '_' + tree_actual_df['Tree no'].astype('str')
    tree_actual_df = tree_actual_df[['tree_id', 'Plot', 'Rep', 'Tree no', 'Hgt22Rod','24_Def', 'Hgt22Drone']]
    tree_actual_df['Hgt22Rod'] = pd.to_numeric(tree_actual_df['Hgt22Rod'], errors='coerce').fillna(0)
    tree_actual_df_no_dead = tree_actual_df[tree_actual_df['24_Def'] != 'D']
    # min_height = tree_actual_df_no_dead['Hgt22Rod'].mean() - sigma * tree_actual_df_no_dead['Hgt22Rod'].std()
    min_height = tree_actual_df_no_dead['Hgt22Rod'].min()

    return tree_actual_df, tree_actual_df_no_dead, min_height
# %%

# %%

# %%

# %%
df2 = df1.copy()
with pd.ExcelWriter('output.xlsx') as writer:  
    df1.to_excel(writer, sheet_name='Sheet_name_1')
    df2.to_excel(writer, sheet_name='Sheet_name_2')
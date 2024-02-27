def test_fun(text):
    print(text)

def plot_history(history):
    try:
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
    except:
        print('No history to plot')


def save_model(model, model_name, src_dir):
    # Save the model to a specified directory
    if CODE_ENV=='local':
        model.save(src_dir + 'models/' + model_name + '.h5')
        
    if CODE_ENV=='kaggle':
        model.save('/kaggle/working/' + model_name + '.h5')

    if CODE_ENV=='aws':
        model.save(src_dir + 'models/' + model_name + '.h5')


def load_model(model_name, src_dir):
    # Start from here if you want to load the model
    # Load the model from a specified directory
    if CODE_ENV=='local':
        model = load_model(src_dir + 'models/' + model_name + '.h5', custom_objects={'rmse': rmse})

    if CODE_ENV=='kaggle':
        model = load_model('/kaggle/input/v1-model/' + model_name + '.h5', custom_objects={'rmse': rmse})

    if CODE_ENV=='aws':
        model = load_model(src_dir + 'models/' + model_name + '.h5', custom_objects={'rmse': rmse})

    return model


# Perform feature engineering
"""
#####- these columns have to be update in the next notebook based on the predictions made by the model #####
- 1 days lag #float 64
- moving average for 7 and 28 days #float 64
- is there a price reduction?
- is there a price increase?
- adjust for inflation?
- consumer sentiment
- holiday
- weather
- 
"""
def feature_engineering(df): 
    ################## lag 1 day sales amount ##############################################################################
    df_conv['sales_amount_lag_1'] = df_conv['sales_amount'].shift(NUM_ITEMS)

    # Now the first value of every entry in the sales_amount_lag_1 column is NaN, because there is no value for the first day of the time series, so we will replace these with the mode (most frequent value) of the sales_amount column
    mode_sales_amount = df_conv.groupby('id')['sales_amount'].agg(lambda x: x.mode()[0])

    # Replace the first day's sales_amount_lag_1 for each item with the mode_sales_amount value
    df_conv.loc[df_conv['d'] == 'd_1', 'sales_amount_lag_1'] = df_conv.loc[df_conv['d'] == 'd_1', 'id'].map(mode_sales_amount)
    ########################################################################################################################

    ################## moving average 7 and 28 days #########################
    df_conv['sales_amount_moving_avg_7'] = df_conv.groupby('id')['sales_amount'].transform(lambda x: x.rolling(window=7).mean())
    df_conv['sales_amount_moving_avg_7'] = df_conv['sales_amount_moving_avg_7'].fillna(method='bfill')

    df_conv['sales_amount_moving_avg_28'] = df_conv.groupby('id')['sales_amount'].transform(lambda x: x.rolling(window=28).mean())
    df_conv['sales_amount_moving_avg_28'] = df_conv['sales_amount_moving_avg_28'].fillna(method='bfill')
    #########################################################################

    ################## days consecutive zero sales and if an entry means that this is a zero sale  #########################
    # Step 1: Mark zero sales days where item is available
    df_conv['zero_sales_available'] = np.where((df_conv['sales_amount'] == 0) & (df_conv['is_available'] == 1), 1, 0).astype(np.int8)

    # Function to apply to each group
    def calculate_consecutive_zeros(group):
        # Step 2: Identify change points to reset the count for consecutive zeros
        group['block'] = (group['zero_sales_available'] == 0).cumsum().astype(np.int16)
        
        # Step 3: Count consecutive zeros within each block
        group['consecutive_zero_sales'] = group.groupby('block').cumcount()
        
        # Reset count where 'zero_sales_available' is 0, as these are not zero sales days or the item is not available
        group['consecutive_zero_sales'] = np.where(group['zero_sales_available'] == 1, group['consecutive_zero_sales'], 0).astype(np.int16)
        
        return group

    # Apply the function to each item group
    df_conv = df_conv.groupby('id', group_keys=False).apply(calculate_consecutive_zeros)

    # Drop the 'block' column because no longer needed
    del df_conv['block']

    return df
########################################################################################################################

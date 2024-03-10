import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def detect_environment():
    # Check for Kaggle
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        print("Environment: kaggle")
        return 'kaggle'
    # Check for AWS
    try:
        response = requests.get('http://169.254.169.254/latest/meta-data/', timeout=2)
        if response.status_code == 200:
            print("Environment: aws")
            return 'aws'
    except requests.exceptions.RequestException:
        pass
    # Default to local
    print("Environment: local")
    return 'local'

def get_paths(env):
    if env=='local':
        ###local###
        #get parent folder of current directory
        parent_dir = '/Users/mf/Desktop/CS/Studies/7_Final_Project/Kaggle_M5PointPrediction'
        #Directory resources
        res_dir = parent_dir + '/res/'
        src_dir = parent_dir + '/src/'
        prc_dir = src_dir + 'processed_data/' # Processed data directory with pickled dataframes
        sub_dir = src_dir + 'submissions/' # Directory to save submission files
        CONFIG_DIR = src_dir + 'models/configs/' # Directory to save model configurations
        CSV_PATH = CONFIG_DIR + 'config.csv'

    if env=='kaggle':
        ###On Kaggle###
        parent_dir = None
        res_dir = '/kaggle/input/m5-forecasting-accuracy/'
        prc_dir = '/kaggle/input/processed-data/'
        src_dir = '/kaggle/working/'
        sub_dir = src_dir + 'submissions/'
        CONFIG_DIR = src_dir + 'configs/'
        CSV_PATH = CONFIG_DIR + 'config.csv'

    if env=='aws':
        parent_dir = '/home/ubuntu/projects/Kaggle_M5PointPrediction'
        res_dir = parent_dir + '/res/'
        src_dir = parent_dir + '/src/'
        prc_dir = src_dir + 'processed_data/' # Processed data directory with pickled dataframes
        sub_dir = src_dir + 'submissions/' # Directory to save submission files
        CONFIG_DIR = src_dir + 'hyperparameter_tuning'
        CSV_PATH = CONFIG_DIR + '/hyperparameter_search_results.csv'
    
    dirs = {
        'parent_dir': parent_dir,
        'res_dir': res_dir,
        'src_dir': src_dir,
        'prc_dir': prc_dir,
        'sub_dir': sub_dir,
        'CONFIG_DIR': CONFIG_DIR,
        'CSV_PATH': CSV_PATH
    }
    return dirs


# Perform feature engineering
"""
#####- these columns have to be update in the next notebook based on the predictions made by the model #####
- 1 days lag #float 64
- moving average for 7 and 28 days #float 64
- ..
"""
def feature_engineering(df, num_block_items): 
    ################## lag 1 day sales amount ##############################################################################
    # After shifting the first days values are NAN but not important as we skip them because we start with the second day
    df['sales_amount_lag_1'] = df['sales_amount'].shift(num_block_items)
    ########################################################################################################################

    ################## moving average 7 and 28 days #########################
    df['sales_amount_moving_avg_7'] = df.groupby('id')['sales_amount'].transform(lambda x: x.rolling(window=7).mean())
    df['sales_amount_moving_avg_7'] = df['sales_amount_moving_avg_7'].fillna(method='bfill')

    df['sales_amount_moving_avg_28'] = df.groupby('id')['sales_amount'].transform(lambda x: x.rolling(window=28).mean())
    df['sales_amount_moving_avg_28'] = df['sales_amount_moving_avg_28'].fillna(method='bfill')
    #########################################################################

    ################# days consecutive zero sales and if an entry means that this is a zero sale  #########################
    # Step 1: Mark zero sales days where item is available
    df['zero_sales_available'] = np.where((df['sales_amount'] == 0) & (df['is_available'] == 1), 1, 0).astype(np.int8)

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
    df = df.groupby('id', group_keys=False).apply(calculate_consecutive_zeros)

    # Drop the 'block' column because no longer needed
    del df['block']

    return df
########################################################################################################################


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

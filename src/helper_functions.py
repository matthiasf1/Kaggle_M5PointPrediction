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

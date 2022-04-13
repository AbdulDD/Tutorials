# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-04-12T15:38:40.586609Z","iopub.execute_input":"2022-04-12T15:38:40.587609Z","iopub.status.idle":"2022-04-12T15:38:40.617618Z","shell.execute_reply.started":"2022-04-12T15:38:40.587473Z","shell.execute_reply":"2022-04-12T15:38:40.616585Z"}}

#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-04-12T15:38:40.618514Z","iopub.status.idle":"2022-04-12T15:38:40.619022Z","shell.execute_reply.started":"2022-04-12T15:38:40.618812Z","shell.execute_reply":"2022-04-12T15:38:40.618835Z"}}
# Let's find Null values
def find_NaN_V(DF):
    Nullvalues = pd.DataFrame({'NaN values': DF.isnull().sum(), 'NaN percent': DF.isnull().mean()*100})
    return Nullvalues


# Lets print columns' name
def Get_colname(DF):
    print('-----------')
    print('Shape: ', DF.shape)
    for col in DF.columns:
        print(col)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-04-12T15:38:40.620364Z","iopub.status.idle":"2022-04-12T15:38:40.620802Z","shell.execute_reply.started":"2022-04-12T15:38:40.620544Z","shell.execute_reply":"2022-04-12T15:38:40.620569Z"}}
# Lets set date time feature

def Datetimeset(DF):
    
    """
    Takes DataFrame and convert sting feature named "date" to datetime feature and also create 
    three saperate columns namely day, month, year.
    """
    
    import datetime as dt
    
    DF['date'] = pd.to_datetime(DF['date'], format='%d.%m.%Y')
    
    DF.sort_values(by = 'date', inplace = True)
    DF['day'] = DF['date'].dt.day
    DF['month'] = DF['date'].dt.month
    DF['year'] = DF['date'].dt.year
    
    DF.drop('date', axis=1, inplace=True)
    
    return DF

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-04-12T15:38:40.628057Z","iopub.execute_input":"2022-04-12T15:38:40.628339Z","iopub.status.idle":"2022-04-12T15:38:40.634625Z","shell.execute_reply.started":"2022-04-12T15:38:40.628308Z","shell.execute_reply":"2022-04-12T15:38:40.633692Z"}}
# Lets do linear regression
def Linear_Reg(Tr_Exmpls, Tr_lbls, Val_Exmpls, Val_lbls):
    
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression().fit(Tr_Exmpls, Tr_lbls)
    print('Training Score:', model.score(Tr_Exmpls, Tr_lbls))
    print('Training Score:', model.score(Val_Exmpls, Val_lbls))

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-04-12T15:38:40.638898Z","iopub.execute_input":"2022-04-12T15:38:40.639169Z","iopub.status.idle":"2022-04-12T15:38:40.648112Z","shell.execute_reply.started":"2022-04-12T15:38:40.639135Z","shell.execute_reply":"2022-04-12T15:38:40.647412Z"}}
# Lets find duplicated values and remove them
def Drop_duplicate_value(DF):
    print('Duplcated observations:', DF.duplicated().sum())
    Droped_value_count = DF.duplicated().sum()
    DF = DF.drop_duplicates()
    
    print('Droped observations:', Droped_value_count)
    print('New Shape:', DF.shape)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-04-12T15:38:40.666618Z","iopub.execute_input":"2022-04-12T15:38:40.667061Z","iopub.status.idle":"2022-04-12T15:38:40.679970Z","shell.execute_reply.started":"2022-04-12T15:38:40.667028Z","shell.execute_reply":"2022-04-12T15:38:40.679314Z"}}
# Find percentage
def Percent(Tr_Observations, Test_Observations):
    
    Tr_Obs_percent = Tr_Observations/(Tr_Observations + Test_Observations)
    Test_Obs_percent = Test_Observations/(Tr_Observations + Test_Observations)
    
    print('No. of Observations (Training Dataset):', Tr_Obs_percent)
    print('No. of Observations (Testing Dataset):', Test_Obs_percent)



#Function to arraning data in form of sklearn dataset
def load_image_files(container_path, Dimension_value):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    
    dimension=(Dimension_value, Dimension_value)
    
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

#def About_Array(Array):
    
#    print('Data type:', Array.dtype)
#    print('Dimension of array:', Array.ndim)
#    print('Itemsize:', Array.itemsize)
#    print('Shape:', Array.shape)
#    print('Memory location:', Array.data)
#    print('Strides:', Array.strides)
#    print('-------------------------------------')
    
def About_Array(Array):
    A_Array = pd.DataFrame({'Dimension of array': Array.ndim,'Data type': Array.dtype,'Shape': Array.shape,'Size': Array.size,'Item size': Array.itemsize,'Number of bytes': Array.nbytes,'Memory location': Array.data,'Strides': Array.strides})
    return A_Array

# %% [markdown]
# # Visualization commands

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}

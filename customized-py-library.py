{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ba5a1904",
   "metadata": {
    "_cell_guid": "d54a4148-48e0-4dcc-aff5-41c61683d905",
    "_uuid": "b0ad977b-e367-42a2-bb09-ddfd51da89de",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-04-12T15:40:57.416752Z",
     "iopub.status.busy": "2022-04-12T15:40:57.414928Z",
     "iopub.status.idle": "2022-04-12T15:40:58.557456Z",
     "shell.execute_reply": "2022-04-12T15:40:58.556518Z",
     "shell.execute_reply.started": "2022-04-12T15:38:40.587473Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.162889,
     "end_time": "2022-04-12T15:40:58.557642",
     "exception": false,
     "start_time": "2022-04-12T15:40:57.394753",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "#Libraries\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import datetime as dt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "3f9bfe53",
   "metadata": {
    "_cell_guid": "56cab4ae-4776-4557-a7e9-e77c5471fccd",
    "_uuid": "9c8d9224-cfc8-4537-a267-24636308a594",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-04-12T15:40:58.589961Z",
     "iopub.status.busy": "2022-04-12T15:40:58.586047Z",
     "iopub.status.idle": "2022-04-12T15:40:58.591842Z",
     "shell.execute_reply": "2022-04-12T15:40:58.592397Z",
     "shell.execute_reply.started": "2022-04-12T15:38:40.618812Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.021069,
     "end_time": "2022-04-12T15:40:58.592568",
     "exception": false,
     "start_time": "2022-04-12T15:40:58.571499",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Let's find Null values\n",
    "def find_NaN_V(DF):\n",
    "    Nullvalues = pd.DataFrame({'NaN values': DF.isnull().sum(), 'NaN percent': DF.isnull().mean()*100})\n",
    "    return Nullvalues\n",
    "\n",
    "\n",
    "# Lets print columns' name\n",
    "def Get_colname(DF):\n",
    "    print('-----------')\n",
    "    print('Shape: ', DF.shape)\n",
    "    for col in DF.columns:\n",
    "        print(col)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8dd0aab8",
   "metadata": {
    "_cell_guid": "8640e4aa-084f-460e-a8d1-0105169f00f7",
    "_uuid": "e691bb96-0c0e-4b3f-acf7-d5373744cbbd",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-04-12T15:40:58.619537Z",
     "iopub.status.busy": "2022-04-12T15:40:58.618711Z",
     "iopub.status.idle": "2022-04-12T15:40:58.625595Z",
     "shell.execute_reply": "2022-04-12T15:40:58.626377Z",
     "shell.execute_reply.started": "2022-04-12T15:38:40.620544Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.022191,
     "end_time": "2022-04-12T15:40:58.626591",
     "exception": false,
     "start_time": "2022-04-12T15:40:58.604400",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Lets set date time feature\n",
    "\n",
    "def Datetimeset(DF):\n",
    "    \n",
    "    \"\"\"\n",
    "    Takes DataFrame and convert sting feature named \"date\" to datetime feature and also create \n",
    "    three saperate columns namely day, month, year.\n",
    "    \"\"\"\n",
    "    \n",
    "    import datetime as dt\n",
    "    \n",
    "    DF['date'] = pd.to_datetime(DF['date'], format='%d.%m.%Y')\n",
    "    \n",
    "    DF.sort_values(by = 'date', inplace = True)\n",
    "    DF['day'] = DF['date'].dt.day\n",
    "    DF['month'] = DF['date'].dt.month\n",
    "    DF['year'] = DF['date'].dt.year\n",
    "    \n",
    "    DF.drop('date', axis=1, inplace=True)\n",
    "    \n",
    "    return DF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e0570576",
   "metadata": {
    "_cell_guid": "432ca933-6267-408c-a453-bbfd98bd3ebf",
    "_uuid": "87d84fda-661c-444d-981f-e73ca74cf21b",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-04-12T15:40:58.653612Z",
     "iopub.status.busy": "2022-04-12T15:40:58.652879Z",
     "iopub.status.idle": "2022-04-12T15:40:58.658706Z",
     "shell.execute_reply": "2022-04-12T15:40:58.657983Z",
     "shell.execute_reply.started": "2022-04-12T15:38:40.628308Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.020405,
     "end_time": "2022-04-12T15:40:58.658850",
     "exception": false,
     "start_time": "2022-04-12T15:40:58.638445",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Lets do linear regression\n",
    "def Linear_Reg(Tr_Exmpls, Tr_lbls, Val_Exmpls, Val_lbls):\n",
    "    \n",
    "    from sklearn.linear_model import LinearRegression\n",
    "    \n",
    "    model = LinearRegression().fit(Tr_Exmpls, Tr_lbls)\n",
    "    print('Training Score:', model.score(Tr_Exmpls, Tr_lbls))\n",
    "    print('Training Score:', model.score(Val_Exmpls, Val_lbls))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11ec8815",
   "metadata": {
    "_cell_guid": "acde63e4-e8bd-47bf-a296-2b8825b28129",
    "_uuid": "42730be2-0824-466d-a77c-e0cba2fc19c4",
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.01162,
     "end_time": "2022-04-12T15:40:58.682136",
     "exception": false,
     "start_time": "2022-04-12T15:40:58.670516",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "04aacdb1",
   "metadata": {
    "_cell_guid": "2a595ba5-f85b-4d58-9371-b4da4c561a87",
    "_uuid": "2d7a6c94-b2fb-406a-9cd8-c8b3535a5b4a",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-04-12T15:40:58.711883Z",
     "iopub.status.busy": "2022-04-12T15:40:58.711162Z",
     "iopub.status.idle": "2022-04-12T15:40:58.713908Z",
     "shell.execute_reply": "2022-04-12T15:40:58.713388Z",
     "shell.execute_reply.started": "2022-04-12T15:38:40.639135Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.02042,
     "end_time": "2022-04-12T15:40:58.714073",
     "exception": false,
     "start_time": "2022-04-12T15:40:58.693653",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Lets find duplicated values and remove them\n",
    "def Drop_duplicate_value(DF):\n",
    "    print('Duplcated observations:', DF.duplicated().sum())\n",
    "    Droped_value_count = DF.duplicated().sum()\n",
    "    DF = DF.drop_duplicates()\n",
    "    \n",
    "    print('Droped observations:', Droped_value_count)\n",
    "    print('New Shape:', DF.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1fea9f45",
   "metadata": {
    "_cell_guid": "c6bd10af-1fda-4e12-9a0d-2dbd5e0e1d08",
    "_uuid": "01196047-90f6-4cbe-9c98-6b80e663b138",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-04-12T15:40:58.742588Z",
     "iopub.status.busy": "2022-04-12T15:40:58.741726Z",
     "iopub.status.idle": "2022-04-12T15:40:58.754113Z",
     "shell.execute_reply": "2022-04-12T15:40:58.754650Z",
     "shell.execute_reply.started": "2022-04-12T15:38:40.667028Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.028238,
     "end_time": "2022-04-12T15:40:58.754855",
     "exception": false,
     "start_time": "2022-04-12T15:40:58.726617",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Find percentage\n",
    "def Percent(Tr_Observations, Test_Observations):\n",
    "    \n",
    "    Tr_Obs_percent = Tr_Observations/(Tr_Observations + Test_Observations)\n",
    "    Test_Obs_percent = Test_Observations/(Tr_Observations + Test_Observations)\n",
    "    \n",
    "    print('No. of Observations (Training Dataset):', Tr_Obs_percent)\n",
    "    print('No. of Observations (Testing Dataset):', Test_Obs_percent)\n",
    "\n",
    "\n",
    "\n",
    "#Function to arraning data in form of sklearn dataset\n",
    "def load_image_files(container_path, Dimension_value):\n",
    "    \"\"\"\n",
    "    Load image files with categories as subfolder names \n",
    "    which performs like scikit-learn sample dataset\n",
    "    \n",
    "    Parameters\n",
    "    ----------\n",
    "    container_path : string or unicode\n",
    "        Path to the main folder holding one subfolder per category\n",
    "    dimension : tuple\n",
    "        size to which image are adjusted to\n",
    "        \n",
    "    Returns\n",
    "    -------\n",
    "    Bunch\n",
    "    \"\"\"\n",
    "    \n",
    "    dimension=(Dimension_value, Dimension_value)\n",
    "    \n",
    "    image_dir = Path(container_path)\n",
    "    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]\n",
    "    categories = [fo.name for fo in folders]\n",
    "\n",
    "    descr = \"A image classification dataset\"\n",
    "    images = []\n",
    "    flat_data = []\n",
    "    target = []\n",
    "    for i, direc in enumerate(folders):\n",
    "        for file in direc.iterdir():\n",
    "            img = imread(file)\n",
    "            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')\n",
    "            flat_data.append(img_resized.flatten()) \n",
    "            images.append(img_resized)\n",
    "            target.append(i)\n",
    "    flat_data = np.array(flat_data)\n",
    "    target = np.array(target)\n",
    "    images = np.array(images)\n",
    "\n",
    "    return Bunch(data=flat_data,\n",
    "                 target=target,\n",
    "                 target_names=categories,\n",
    "                 images=images,\n",
    "                 DESCR=descr)\n",
    "\n",
    "def About_Array(Array):\n",
    "    \n",
    "    print('Data type:', Array.dtype)\n",
    "    print('Itemsize:', Array.itemsize)\n",
    "    print('Shape:', Array.shape)\n",
    "    print('Memory location:', Array.data)\n",
    "    print('Strides:', Array.strides)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f938688a",
   "metadata": {
    "_cell_guid": "3d046531-221b-4728-8c8b-a70e6b72b702",
    "_uuid": "65e08ec4-8941-42b4-bf0e-38b19220f304",
    "papermill": {
     "duration": 0.012062,
     "end_time": "2022-04-12T15:40:58.778865",
     "exception": false,
     "start_time": "2022-04-12T15:40:58.766803",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Visualization commands"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d5ac7cd3",
   "metadata": {
    "_cell_guid": "c5a350ec-4912-4020-afe3-5b69d4850cc3",
    "_uuid": "95202509-5fc9-42f5-864d-109fd892e526",
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.011378,
     "end_time": "2022-04-12T15:40:58.802335",
     "exception": false,
     "start_time": "2022-04-12T15:40:58.790957",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a26f512a",
   "metadata": {
    "_cell_guid": "f39b9de3-d824-4c55-a98a-38be9f24f0bc",
    "_uuid": "abbc1bda-12f8-43c5-898b-5c8dde202c5f",
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.0113,
     "end_time": "2022-04-12T15:40:58.825415",
     "exception": false,
     "start_time": "2022-04-12T15:40:58.814115",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 12.871775,
   "end_time": "2022-04-12T15:40:59.548149",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2022-04-12T15:40:46.676374",
   "version": "2.3.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

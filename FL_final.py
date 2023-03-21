# import important libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import nest_asyncio
nest_asyncio.apply()
import tensorflow_federated as tff


# setting options to show all columns
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

# Read in the dataset in csv
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Federated Learning/kidney_disease.csv')
df = df.drop('id', axis=1)
df.head()

# Assess the data
df.shape
df.info()

# Data Cleaning
# Cleaning the column 'rc'
df['rc'][162] = 4.8
df['rc'] = df['rc'].astype(float)
rc_float = rc_not_float.astype(float)
type(df['rc'][382])     #Check 

# Cleaning the colunmn 'pcv'
df['pcv'][66] = 48
df['pcv'][214] = 43
df['pcv'] = df['pcv'].astype(float)

# Cleaning the colunmn 'wc'
# Correcting values in indexes, 185, 133, 76
df['wc'][[76, 133, 185]] = [6200, 8400, 5800]
df['wc'] = df['wc'].astype(float)

# Filling missing values of numeric categorical columns with previous values
nom_cat_col = ['sg', 'al', 'su']

for col in nom_cat_col:
  df[col] = df[col].fillna(method = 'ffill')

df.info()


# Filling the missing values of numerical variables with their mean.
numerical_col = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

for col in numerical_col:
  df[col] = df[col].fillna(df[col].mean())

df.info()


# filling the missing values of categorical variables with 'unknown'.
cat_col = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

for col in cat_col:
  df[col] = df[col].fillna('unknown')

df.info()


# Changing values of target variable (classification) to 1 and 0
df[df['classification']=='ckd\t']['classification']     # checking the inappropriate values
df['classification'][[37, 230]] = 'ckd'                 # fixing inappropriate data
label = {'ckd':1, 'notckd':0}                           
df['classification'].replace(label, inplace=True)       # Replace values with 1 and 0
df['classification'].value_counts()                     # Check

#### DATA PREPARATION

x = df.drop('classification', axis = 1)
y = df['classification']

# create a list of continous columns and categorical columns
x_numeric = x[['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 
               'pot', 'hemo', 'pcv', 'wc', 'rc']]
x_cat = x[['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
           'htn', 'dm', 'cad', 'appet', 'pe', 'ane']]


# Standardize the numeric data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_standardized = ss.fit_transform(x_numeric)

# Continued Preprocessing
# Create two input layers for tensorflow model, for both numerical and categorical feature   
numeric_inputs = tf.keras.layers.Input((11,), name='numeric_inputs')
cat_inputs = tf.keras.layers.Input((13,), name='cat_inputs')

# The categorical layers are fed into an embedded layer
for i in x_cat.columns:
  print(x_cat[i].nunique())

num_categories = 51

  def emb_sz_rule(n_cat): 
      return min(600, round(1.6 * n_cat**0.56))

  embedding_layer = tf.keras.layers.Embedding(num_categories, 
      emb_sz_rule(num_categories), 
      input_length=13)
  cats = embedding_layer(cat_inputs)
  cats = tf.keras.layers.Flatten()(cats)

# Concatenate the preprocessed numerical and categorical data to make your final feature dataset
df = tf.keras.layers.Concatenate()([cats, numeric_inputs])


# MAKE DATASET INTO A FEDERATED DATASET

client_id_colname = 'client_num' # the column that represents client ID
SHUFFLE_BUFFER = 100
NUM_EPOCHS = 1
# split client id into train and test clients
client_ids = df[client_id_colname].unique()
train_client_ids = pd.DataFrame(client_ids).sample(frac=0.5).values.tolist()
test_client_ids = [y for y in client_ids if y not in train_client_ids]

def create_tf_dataset_for_client_fn(client_id):
  # a function which takes a client_id and returns a
  # tf.data.Dataset for that client
  client_data = df[df[client_id_colname] == client_id[0]]
  dataset = tf.data.Dataset.from_tensor_slices(client_data.fillna('').to_dict("list"))
  dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(1).repeat(NUM_EPOCHS)
  return dataset

train_data = tff.simulation.datasets.ClientData.from_clients_and_fn(
        client_ids=train_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )
test_data = tff.simulation.datasets.ClientData.from_clients_and_fn(
        client_ids=test_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )
example_dataset = train_data.create_tf_dataset_for_client(
        train_data.client_ids[0]
    )


### MODEL BUILDING
# The model is built as a federated learning model, using the TFF API,
# It can learn on other devices and aggregate the training metrics for each client 
def create_keras_model():
  
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(df.shape[0])),
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=df.element_spec,
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[tf.keras.metrics.Accuracy()])

# Federated algorithms have 4 main components
#   A server to client broadcast
#   A local client update step
#   A client-to-server upload step
#   A server update step

# Federated Averaging of the client model metrics
# Done on the server using the TFF API
train = tff.learning.build_federated_averaging_process(model_fn)

# the initialize() method retrieves the initial server state
state = train.initialize()

"""
train.next is called which will run our federated training. 
This includes sending the initial server state to each of the clients.
Each client will run its own local rounds of training in this case 20 rounds
and then send an update to the server. The server stores the new aggregated 
global model produced from the decentralized data.
"""

for _ in range (20):
  state, metrics = train.next(state, train_data)
  print (metrics.loss) 


# Finally, we can perform federated evaluation to understand
# the state of our trained model

eval = tff.learning.build_federated_evaluation(model_fn)
metrics = eval(state.model, test_data)
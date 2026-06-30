#!/usr/bin/env python
# coding: utf-8

# # Ungraded Lab: Hyperparameter tuning and model training with TFX
# 
# In this lab, you will be again doing hyperparameter tuning but this time, it will be within a [Tensorflow Extended (TFX)](https://www.tensorflow.org/tfx/) pipeline. 
# 
# We have already introduced some TFX components in Course 2 of this specialization related to data ingestion, validation, and transformation. In this notebook, you will get to work with two more which are related to model development and training: *Tuner* and *Trainer*.
# 
# <img src='https://www.tensorflow.org/tfx/guide/images/prog_trainer.png' alt='tfx pipeline'>
# image source: https://www.tensorflow.org/tfx/guide
# 
# * The *Tuner* utilizes the [Keras Tuner](https://keras-team.github.io/keras-tuner/) API under the hood to tune your model's hyperparameters.
# * You can get the best set of hyperparameters from the Tuner component and feed it into the *Trainer* component to optimize your model for training.
# 
# You will again be working with the [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset and will feed it though the TFX pipeline up to the Trainer component.You will quickly review the earlier components from Course 2, then focus on the two new components introduced.
# 
# Let's begin!
# 
# 

# ## Imports

# In[ ]:


import os
import pprint

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from absl import logging

from tfx import v1 as tfx
from tfx.proto import example_gen_pb2, trainer_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

tf.get_logger().propagate = False
tf.get_logger().setLevel('ERROR')
pp = pprint.PrettyPrinter()
logging.set_verbosity(logging.ERROR)


# ## Load and prepare the dataset
# 
# As mentioned earlier, you will be using the Fashion MNIST dataset just like in the previous lab. This will allow you to compare the similarities and differences when using Keras Tuner as a standalone library and within an ML pipeline.
# 
# You will first need to setup the directories that you will use to store the dataset, as well as the pipeline artifacts and metadata store.

# In[ ]:


# Location of the pipeline metadata store
_pipeline_root = './pipeline/'

# Directory of the raw data files
_data_root = './data/fmnist'

# Temporary directory
tempdir = './tempdir'


# In[ ]:


# Create the dataset directory
get_ipython().system('mkdir -p {_data_root}')

# Create the TFX pipeline files directory
get_ipython().system('mkdir {_pipeline_root}')


# You will now load FashionMNIST from [Tensorflow Datasets](https://www.tensorflow.org/datasets). The `with_info` flag will be set to `True` so you can display information about the dataset in the next cell (i.e. using `ds_info`). This is already in your workspace so the `download` flag is set to `False`.

# In[ ]:


# Download the dataset
ds, ds_info = tfds.load('fashion_mnist', data_dir=tempdir, with_info=True, download=False)


# In[ ]:


# Display info about the dataset
print(ds_info)


# You can review the downloaded files with the code below. For this lab, you will be using the *train* TFRecord so you will need to take note of its filename. You will not use the *test* TFRecord in this lab.

# In[ ]:


# Define the location of the train tfrecord downloaded via TFDS
tfds_data_path = f'{tempdir}/{ds_info.name}/{ds_info.version}'

# Display contents of the TFDS data directory
os.listdir(tfds_data_path)


# You will then copy the train split from the downloaded data so it can be consumed by the ExampleGen component in the next step. This component requires that your files are in a directory without extra files (e.g. JSONs and TXT files).

# In[ ]:


# Define the train tfrecord filename
train_filename = 'fashion_mnist-train.tfrecord-00000-of-00001'

# Copy the train tfrecord into the data root folder
get_ipython().system('cp {tfds_data_path}/{train_filename} {_data_root}')


# ## TFX Pipeline
# 
# With the setup complete, you can now proceed to creating the pipeline. 

# ### Initialize the Interactive Context
# 
# You will start by initializing the [InteractiveContext](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/experimental/interactive/interactive_context.py) so you can run the components within this notebook environment. You can safely ignore the warning because you will just be using a local SQLite file for the metadata store.

# In[ ]:


# Initialize the InteractiveContext
context = InteractiveContext(pipeline_root=_pipeline_root)


# ### ExampleGen
# 
# You will start the pipeline by ingesting the TFRecord you set aside. The [ImportExampleGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/components/ImportExampleGen) consumes TFRecords and you can specify splits as shown below. For this exercise, you will split the train tfrecord to use 80% for the train set, and the remaining 20% as eval/validation set.

# In[ ]:


# Specify 80/20 split for the train and eval set
output = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
        example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
    ]))

# Ingest the data through ExampleGen
example_gen = tfx.components.ImportExampleGen(input_base=_data_root, output_config=output)

# Run the component
context.run(example_gen)


# In[ ]:


# Print split names and URI
artifact = example_gen.outputs['examples'].get()[0]
print(artifact.split_names, artifact.uri)


# ### StatisticsGen
# 
# Next, you will compute the statistics of the dataset with the [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) component.

# In[ ]:


# Run StatisticsGen
statistics_gen = tfx.components.StatisticsGen(
    examples=example_gen.outputs['examples'])

context.run(statistics_gen)


# ### SchemaGen
# 
# You can then infer the dataset schema with [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen). This will be used to validate incoming data to ensure that it is formatted correctly.

# In[ ]:


# Run SchemaGen
schema_gen = tfx.components.SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
context.run(schema_gen)


# In[ ]:


# Visualize the results
context.show(schema_gen.outputs['schema'])


# ### ExampleValidator
# 
# You can assume that the dataset is clean since we downloaded it from TFDS. But just to review, let's run it through [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) to detect if there are anomalies within the dataset.

# In[ ]:


# Run ExampleValidator
example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])
context.run(example_validator)


# In[ ]:


# Visualize the results. There should be no anomalies.
context.show(example_validator.outputs['anomalies'])


# ### Transform
# 
# Let's now use the [Transform](https://www.tensorflow.org/tfx/guide/transform) component to scale the image pixels and convert the data types to float. You will first define the transform module containing these operations before you run the component.

# In[ ]:


_transform_module_file = 'fmnist_transform.py'


# In[ ]:


get_ipython().run_cell_magic('writefile', '{_transform_module_file}', '\nimport tensorflow as tf\nimport tensorflow_transform as tft\n\n# Keys\n_LABEL_KEY = \'label\'\n_IMAGE_KEY = \'image\'\n\n\ndef _transformed_name(key):\n    return key + \'_xf\'\n\ndef _image_parser(image_str):\n    \'\'\'converts the images to a float tensor\'\'\'\n    image = tf.image.decode_image(image_str, channels=1)\n    image = tf.reshape(image, (28, 28, 1))\n    image = tf.cast(image, tf.float32)\n    return image\n\n\ndef _label_parser(label_id):\n    \'\'\'converts the labels to a float tensor\'\'\'\n    label = tf.cast(label_id, tf.float32)\n    return label\n\n\ndef preprocessing_fn(inputs):\n    """tf.transform\'s callback function for preprocessing inputs.\n    Args:\n        inputs: map from feature keys to raw not-yet-transformed features.\n    Returns:\n        Map from string feature key to transformed feature operations.\n    """\n    \n    # Convert the raw image and labels to a float array\n    with tf.device("/cpu:0"):\n        outputs = {\n            _transformed_name(_IMAGE_KEY):\n                tf.map_fn(\n                    _image_parser,\n                    tf.squeeze(inputs[_IMAGE_KEY], axis=1),\n                    dtype=tf.float32),\n            _transformed_name(_LABEL_KEY):\n                tf.map_fn(\n                    _label_parser,\n                    inputs[_LABEL_KEY],\n                    dtype=tf.float32)\n        }\n    \n    # scale the pixels from 0 to 1\n    outputs[_transformed_name(_IMAGE_KEY)] = tft.scale_to_0_1(outputs[_transformed_name(_IMAGE_KEY)])\n    \n    return outputs')


# You will run the component by passing in the examples, schema, and transform module file.
# 
# *Note: You can safely ignore the warnings and `udf_utils` related errors.*

# In[ ]:


# Setup the Transform component
transform = tfx.components.Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_transform_module_file))

# Run the component
context.run(transform)


# ### Tuner
# 
# As the name suggests, the [Tuner](https://www.tensorflow.org/tfx/guide/tuner) component tunes the hyperparameters of your model. To use this, you will need to provide a *tuner module file* which contains a `tuner_fn()` function. In this function, you will mostly do the same steps as you did in the previous ungraded lab but with some key differences in handling the dataset. 
# 
# The Transform component earlier saved the transformed examples as TFRecords compressed in `.gz` format and you will need to load that into memory. Once loaded, you will need to create batches of features and labels so you can finally use it for hypertuning. This process is modularized in the `_input_fn()` below. 
# 
# Going back, the `tuner_fn()` function will return a `TunerFnResult` [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple) containing your `tuner` object and a set of arguments to pass to `tuner.search()` method. You will see these in action in the following cells. When reviewing the module file, we recommend viewing the `tuner_fn()` first before looking at the other auxiliary functions.

# In[ ]:


# Declare name of module file
_tuner_module_file = 'tuner.py'


# In[ ]:


get_ipython().run_cell_magic('writefile', '{_tuner_module_file}', '\n# Define imports\nfrom kerastuner.engine import base_tuner\nimport kerastuner as kt\nfrom tensorflow import keras\nfrom typing import NamedTuple, Dict, Text, Any, List\nfrom tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor\nimport tensorflow as tf\nimport tensorflow_transform as tft\n\n# Declare namedtuple field names\nTunerFnResult = NamedTuple(\'TunerFnResult\', [(\'tuner\', base_tuner.BaseTuner),\n                                             (\'fit_kwargs\', Dict[Text, Any])])\n\n# Input key\n_IMAGE_KEY = \'image_xf\'\n\n# Label key\n_LABEL_KEY = \'label_xf\'\n\n# Callback for the search strategy\nstop_early = tf.keras.callbacks.EarlyStopping(monitor=\'val_loss\', patience=5)\n\n\ndef _gzip_reader_fn(filenames):\n  \'\'\'Load compressed dataset\n  \n  Args:\n    filenames - filenames of TFRecords to load\n\n  Returns:\n    TFRecordDataset loaded from the filenames\n  \'\'\'\n\n  # Load the dataset. Specify the compression type since it is saved as `.gz`\n  return tf.data.TFRecordDataset(filenames, compression_type=\'GZIP\')\n  \n\ndef _input_fn(file_pattern,\n              tf_transform_output,\n              num_epochs=None,\n              batch_size=32) -> tf.data.Dataset:\n  \'\'\'Create batches of features and labels from TF Records\n\n  Args:\n    file_pattern - List of files or patterns of file paths containing Example records.\n    tf_transform_output - transform output graph\n    num_epochs - Integer specifying the number of times to read through the dataset. \n            If None, cycles through the dataset forever.\n    batch_size - An int representing the number of records to combine in a single batch.\n\n  Returns:\n    A dataset of dict elements, (or a tuple of dict elements and label). \n    Each dict maps feature keys to Tensor or SparseTensor objects.\n  \'\'\'\n\n  # Get feature specification based on transform output\n  transformed_feature_spec = (\n      tf_transform_output.transformed_feature_spec().copy())\n  \n  # Create batches of features and labels\n  dataset = tf.data.experimental.make_batched_features_dataset(\n      file_pattern=file_pattern,\n      batch_size=batch_size,\n      features=transformed_feature_spec,\n      reader=_gzip_reader_fn,\n      num_epochs=num_epochs,\n      label_key=_LABEL_KEY)\n  \n  return dataset\n\n\ndef model_builder(hp):\n  \'\'\'\n  Builds the model and sets up the hyperparameters to tune.\n\n  Args:\n    hp - Keras tuner object\n\n  Returns:\n    model with hyperparameters to tune\n  \'\'\'\n\n  # Initialize the Sequential API and start stacking the layers\n  model = keras.Sequential()\n  model.add(keras.layers.Input(shape=(28, 28, 1), name=_IMAGE_KEY))\n  model.add(keras.layers.Flatten())\n\n  # Tune the number of units in the first Dense layer\n  # Choose an optimal value between 32-512\n  hp_units = hp.Int(\'units\', min_value=32, max_value=512, step=32)\n  model.add(keras.layers.Dense(units=hp_units, activation=\'relu\', name=\'dense_1\'))\n\n  # Add next layers\n  model.add(keras.layers.Dropout(0.2))\n  model.add(keras.layers.Dense(10, activation=\'softmax\'))\n\n  # Tune the learning rate for the optimizer\n  # Choose an optimal value from 0.01, 0.001, or 0.0001\n  hp_learning_rate = hp.Choice(\'learning_rate\', values=[1e-2, 1e-3, 1e-4])\n\n  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),\n                loss=keras.losses.SparseCategoricalCrossentropy(),\n                metrics=[\'accuracy\'])\n\n  return model\n\ndef tuner_fn(fn_args: FnArgs) -> TunerFnResult:\n  """Build the tuner using the KerasTuner API.\n  Args:\n    fn_args: Holds args as name/value pairs.\n\n      - working_dir: working dir for tuning.\n      - train_files: List of file paths containing training tf.Example data.\n      - eval_files: List of file paths containing eval tf.Example data.\n      - train_steps: number of train steps.\n      - eval_steps: number of eval steps.\n      - schema_path: optional schema of the input data.\n      - transform_graph_path: optional transform graph produced by TFT.\n  \n  Returns:\n    A namedtuple contains the following:\n      - tuner: A BaseTuner that will be used for tuning.\n      - fit_kwargs: Args to pass to tuner\'s run_trial function for fitting the\n                    model , e.g., the training and validation dataset. Required\n                    args depend on the above tuner\'s implementation.\n  """\n\n  # Define tuner search strategy\n  tuner = kt.Hyperband(model_builder,\n                     objective=\'val_accuracy\',\n                     max_epochs=10,\n                     factor=3,\n                     directory=fn_args.working_dir,\n                     project_name=\'kt_hyperband\')\n\n  # Load transform output\n  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)\n\n  # Use _input_fn() to extract input features and labels from the train and val set\n  train_set = _input_fn(fn_args.train_files[0], tf_transform_output)\n  val_set = _input_fn(fn_args.eval_files[0], tf_transform_output)\n\n\n  return TunerFnResult(\n      tuner=tuner,\n      fit_kwargs={ \n          "callbacks":[stop_early],\n          \'x\': train_set,\n          \'validation_data\': val_set,\n          \'steps_per_epoch\': fn_args.train_steps,\n          \'validation_steps\': fn_args.eval_steps\n      }\n  )')


# With the module defined, you can now setup the Tuner component. You can see the description of each argument [here](https://www.tensorflow.org/tfx/api_docs/python/tfx/components/Tuner). 
# 
# Notice that we passed a `num_steps` argument to the train and eval args and this was used in the `steps_per_epoch` and `validation_steps` arguments in the tuner module above. This can be useful if you don't want to go through the entire dataset when tuning. For example, if you have 10GB of training data, it would be incredibly time consuming if you will iterate through it entirely just for one epoch and one set of hyperparameters. You can set the number of steps so your program will only go through a fraction of the dataset. 
# 
# You can compute for the total number of steps in one epoch by: `number of examples / batch size`. For this particular example, we have `48000 examples / 32 (default size)` which equals `1500` steps per epoch for the train set (compute val steps from 12000 examples). Since you passed `500` in the `num_steps` of the train args, this means that some examples will be skipped. This will likely result in lower accuracy readings but will save time in doing the hypertuning. Try modifying this value later and see if you arrive at the same set of hyperparameters.

# In[ ]:


# Setup the Tuner component
tuner = tfx.components.Tuner(
    module_file=_tuner_module_file,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=500),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=100)
    )


# In[ ]:


# Run the component. This will take around 10 minutes to run.
# When done, it will summarize the results and show the 10 best trials.
context.run(tuner, enable_cache=False)


# ### Trainer
# 
# Like the Tuner component, the [Trainer](https://www.tensorflow.org/tfx/guide/trainer) component also requires a module file to setup the training process. It will look for a `run_fn()` function that defines and trains the model. The steps will look similar to the tuner module file:
# 
# * Define the model - You can get the results of the Tuner component through the `fn_args.hyperparameters` argument. You will see it passed into the `model_builder()` function below. If you didn't run `Tuner`, then you can just explicitly define the number of hidden units and learning rate.
# 
# * Load the train and validation sets - You have done this in the Tuner component. For this module, you will pass in a `num_epochs` value (10) to indicate how many batches will be prepared. You can opt not to do this and pass a `num_steps` value as before.
# 
# * Setup and train the model - This will look very familiar if you're already used to the [Keras Models Training API](https://keras.io/api/models/model_training_apis/). You can pass in callbacks like the [TensorBoard callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard) so you can visualize the results later.
# 
# * Save the model - This is needed so you can analyze and serve your model. You will get to do this in later parts of the course and specialization.

# In[ ]:


# Declare trainer module file
_trainer_module_file = 'trainer.py'


# In[ ]:


get_ipython().run_cell_magic('writefile', '{_trainer_module_file}', '\nfrom tensorflow import keras\nfrom typing import NamedTuple, Dict, Text, Any, List\nfrom tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor\nimport tensorflow as tf\nimport tensorflow_transform as tft\n\n# Input key\n_IMAGE_KEY = \'image_xf\'\n\n# Label key\n_LABEL_KEY = \'label_xf\'\n\ndef _gzip_reader_fn(filenames):\n  \'\'\'Load compressed dataset\n  \n  Args:\n    filenames - filenames of TFRecords to load\n\n  Returns:\n    TFRecordDataset loaded from the filenames\n  \'\'\'\n\n  # Load the dataset. Specify the compression type since it is saved as `.gz`\n  return tf.data.TFRecordDataset(filenames, compression_type=\'GZIP\')\n  \n\ndef _input_fn(file_pattern,\n              tf_transform_output,\n              num_epochs=None,\n              batch_size=32) -> tf.data.Dataset:\n  \'\'\'Create batches of features and labels from TF Records\n\n  Args:\n    file_pattern - List of files or patterns of file paths containing Example records.\n    tf_transform_output - transform output graph\n    num_epochs - Integer specifying the number of times to read through the dataset. \n            If None, cycles through the dataset forever.\n    batch_size - An int representing the number of records to combine in a single batch.\n\n  Returns:\n    A dataset of dict elements, (or a tuple of dict elements and label). \n    Each dict maps feature keys to Tensor or SparseTensor objects.\n  \'\'\'\n  transformed_feature_spec = (\n      tf_transform_output.transformed_feature_spec().copy())\n  \n  dataset = tf.data.experimental.make_batched_features_dataset(\n      file_pattern=file_pattern,\n      batch_size=batch_size,\n      features=transformed_feature_spec,\n      reader=_gzip_reader_fn,\n      num_epochs=num_epochs,\n      label_key=_LABEL_KEY)\n  \n  return dataset\n\n\ndef model_builder(hp):\n  \'\'\'\n  Builds the model and sets up the hyperparameters to tune.\n\n  Args:\n    hp - Keras tuner object\n\n  Returns:\n    model with hyperparameters to tune\n  \'\'\'\n\n  # Initialize the Sequential API and start stacking the layers\n  model = keras.Sequential()\n  model.add(keras.layers.Input(shape=(28, 28, 1), name=_IMAGE_KEY))\n  model.add(keras.layers.Flatten())\n\n  # Get the number of units from the Tuner results\n  hp_units = hp.get(\'units\')\n  model.add(keras.layers.Dense(units=hp_units, activation=\'relu\'))\n\n  # Add next layers\n  model.add(keras.layers.Dropout(0.2))\n  model.add(keras.layers.Dense(10, activation=\'softmax\'))\n\n  # Get the learning rate from the Tuner results\n  hp_learning_rate = hp.get(\'learning_rate\')\n\n  # Setup model for training\n  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),\n                loss=keras.losses.SparseCategoricalCrossentropy(),\n                metrics=[\'accuracy\'])\n\n  # Print the model summary\n  model.summary()\n  \n  return model\n\n\ndef run_fn(fn_args: FnArgs) -> None:\n  """Defines and trains the model.\n  Args:\n    fn_args: Holds args as name/value pairs. Refer here for the complete attributes: \n    https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs#attributes\n  """\n  \n  # Load transform output\n  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)\n  \n  # Create batches of data good for 10 epochs\n  train_set = _input_fn(fn_args.train_files[0], tf_transform_output, 10)\n  val_set = _input_fn(fn_args.eval_files[0], tf_transform_output, 10)\n\n  # Load best hyperparameters\n  hp = fn_args.hyperparameters.get(\'values\')\n\n  # Build the model\n  model = model_builder(hp)\n\n  # Train the model\n  model.fit(\n      x=train_set,\n      validation_data=val_set,\n      )\n  \n  # Save the model\n  model.save(fn_args.serving_model_dir, save_format=\'tf\')')


# You can pass the output of the `Tuner` component to the `Trainer` by filling the `hyperparameters` argument with the `Tuner` output. This is indicated by the `tuner.outputs['best_hyperparameters']` below. You can see the definition of the other arguments [here](https://www.tensorflow.org/tfx/api_docs/python/tfx/components/Trainer).

# In[ ]:


# Setup the Trainer component
trainer = tfx.components.Trainer(
    module_file=_trainer_module_file,
    examples=transform.outputs['transformed_examples'],
    hyperparameters=tuner.outputs['best_hyperparameters'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval']))


# Take note that when re-training your model, you don't always have to retune your hyperparameters. Once you have a set that you think performs well, you can just import it with the `Importer` component as shown in the [official docs](https://www.tensorflow.org/tfx/guide/tuner):
# 
# ```
# hparams_importer = Importer(
#     # This can be Tuner's output file or manually edited file. The file contains
#     # text format of hyperparameters (keras_tuner.HyperParameters.get_config())
#     source_uri='path/to/best_hyperparameters.txt',
#     artifact_type=HyperParameters,
# ).with_id('import_hparams')
# 
# trainer = Trainer(
#     ...
#     # An alternative is directly use the tuned hyperparameters in Trainer's user
#     # module code and set hyperparameters to None here.
#     hyperparameters = hparams_importer.outputs['result'])
# ```

# In[ ]:


# Run the component
context.run(trainer, enable_cache=False)


# Your model should now be saved in your pipeline directory and you can navigate through it as shown below. The file is saved as `saved_model.pb`.

# In[ ]:


# Get artifact uri of trainer model output
model_artifact_dir = trainer.outputs['model'].get()[0].uri

# List subdirectories artifact uri
print(f'contents of model artifact directory:{os.listdir(model_artifact_dir)}')

# Define the model directory
model_dir = os.path.join(model_artifact_dir, 'Format-Serving')

# List contents of model directory
print(f'contents of model directory: {os.listdir(model_dir)}')


# ***Congratulations! You have now created an ML pipeline that includes hyperparameter tuning and model training. You will know more about the next components in future lessons but in the next section, you will first learn about a framework for automatically building ML pipelines: AutoML. Enjoy the rest of the course!***

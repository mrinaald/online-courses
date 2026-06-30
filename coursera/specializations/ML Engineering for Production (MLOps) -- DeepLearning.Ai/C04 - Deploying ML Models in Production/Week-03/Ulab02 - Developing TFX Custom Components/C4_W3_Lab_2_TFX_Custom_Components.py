#!/usr/bin/env python
# coding: utf-8

# # Ungraded Lab: Developing Custom TFX Components
# 
# [Tensorflow Extended (TFX)](https://www.tensorflow.org/tfx/) provides ready made components for typical steps in a machine learning workflow. Other courses in this specialization focus on specific components and in this lab, you will learn how to make your own. This will be useful in case your project has specific needs that fall outside the standard TFX components. It will make your pipelines more flexible while still leveraging the experiment tracking and orchestration that TFX provides. In particular, you will:
# 
# * build a custom component using Python functions
# * build a custom component by reusing an existing TFX component
# * run a TFX pipeline locally using a built-in pipeline orchestrator
# 
# To demonstrate, you will run the pipeline used in this [official tutorial](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/tfx/penguin_simple.ipynb) then modify it to have a custom component. Some of the discussions here are also taken from that tutorial to explain the motivation and point to additional resources.
# 
# Let's begin!
# 
# *Note: If you haven't taken other courses in this specialization and it's the first time you're using TFX, please see
# [Understanding TFX Pipelines](https://www.tensorflow.org/tfx/guide/understanding_tfx_pipelines)
# to get an overview of important concepts.*

# ## Imports

# In[ ]:


import os
from absl import logging
import pandas as pd
import glob

import tensorflow as tf
print('TensorFlow version: {}'.format(tf.__version__))
from tfx import v1 as tfx
print('TFX version: {}'.format(tfx.__version__))

# Set default logging level.
logging.set_verbosity(logging.INFO) 


# ### Set up variables
# 
# There are some variables used to define a pipeline. You can customize these
# variables as you want. By default all output from the pipeline will be
# generated under the current directory.

# In[ ]:


# Pipeline label
PIPELINE_NAME = "penguin-simple"

# Output directory to store artifacts generated from the pipeline.
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)

# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

# Dataset directory
DATA_ROOT = 'data'


# ### Dataset
# 
# You will use the
# [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/articles/intro.html) for this exercise. It has four numeric features, namely:
# 
# - culmen_length_mm
# - culmen_depth_mm
# - flipper_length_mm
# - body_mass_g
# 
# All features were already normalized to have range [0,1]. You will build a
# classification model which predicts the `species` of penguins. 

# ## Running the pipeline using standard components
# 
# The pipeline will consist of three essential TFX components and the graph will look like this:
# 
# ```
# ExampleGen -> Trainer -> Pusher
# ``` 
# 
# The pipeline includes the most minimal ML workflow which is
# importing data (ExampleGen), training a model (Trainer) and exporting the trained model (Pusher).

# ### Model training code
# 
# You will first define the trainer module so the `Trainer` component can build the model and train it.

# In[ ]:


_trainer_module_file = 'penguin_trainer.py'


# In[ ]:


get_ipython().run_cell_magic('writefile', '{_trainer_module_file}', '\nfrom typing import List\nfrom absl import logging\nimport tensorflow as tf\nfrom tensorflow import keras\nfrom tensorflow_transform.tf_metadata import schema_utils\n\nfrom tfx import v1 as tfx\nfrom tfx_bsl.public import tfxio\nfrom tensorflow_metadata.proto.v0 import schema_pb2\n\n_FEATURE_KEYS = [\n    \'culmen_length_mm\', \'culmen_depth_mm\', \'flipper_length_mm\', \'body_mass_g\'\n]\n_LABEL_KEY = \'species\'\n\n_TRAIN_BATCH_SIZE = 20\n_EVAL_BATCH_SIZE = 10\n\n# Since we\'re not generating or creating a schema, we will instead create\n# a feature spec.  Since there are a fairly small number of features this is\n# manageable for this dataset.\n_FEATURE_SPEC = {\n    **{\n        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)\n           for feature in _FEATURE_KEYS\n       },\n    _LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)\n}\n\n\ndef _input_fn(file_pattern: List[str],\n              data_accessor: tfx.components.DataAccessor,\n              schema: schema_pb2.Schema,\n              batch_size: int = 200) -> tf.data.Dataset:\n  """Generates features and label for training.\n\n  Args:\n    file_pattern: List of paths or patterns of input tfrecord files.\n    data_accessor: DataAccessor for converting input to RecordBatch.\n    schema: schema of the input data.\n    batch_size: representing the number of consecutive elements of returned\n      dataset to combine in a single batch\n\n  Returns:\n    A dataset that contains (features, indices) tuple where features is a\n      dictionary of Tensors, and indices is a single Tensor of label indices.\n  """\n  return data_accessor.tf_dataset_factory(\n      file_pattern,\n      tfxio.TensorFlowDatasetOptions(\n          batch_size=batch_size, label_key=_LABEL_KEY),\n      schema=schema).repeat()\n\n\ndef _build_keras_model() -> tf.keras.Model:\n  """Creates a DNN Keras model for classifying penguin data.\n\n  Returns:\n    A Keras Model.\n  """\n  # The model below is built with Functional API, please refer to\n  # https://www.tensorflow.org/guide/keras/overview for all API options.\n  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]\n  d = keras.layers.concatenate(inputs)\n  for _ in range(2):\n    d = keras.layers.Dense(8, activation=\'relu\')(d)\n  outputs = keras.layers.Dense(3)(d)\n\n  model = keras.Model(inputs=inputs, outputs=outputs)\n  model.compile(\n      optimizer=keras.optimizers.Adam(1e-2),\n      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\n      metrics=[keras.metrics.SparseCategoricalAccuracy()])\n\n  model.summary(print_fn=logging.info)\n  return model\n\n\n# TFX Trainer will call this function.\ndef run_fn(fn_args: tfx.components.FnArgs):\n  """Train the model based on given args.\n\n  Args:\n    fn_args: Holds args used to train the model as name/value pairs.\n  """\n\n  # This schema is usually either an output of SchemaGen or a manually-curated\n  # version provided by pipeline author. A schema can also derived from TFT\n  # graph if a Transform component is used. In the case when either is missing,\n  # `schema_from_feature_spec` could be used to generate schema from very simple\n  # feature_spec, but the schema returned would be very primitive.\n  schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)\n\n  train_dataset = _input_fn(\n      fn_args.train_files,\n      fn_args.data_accessor,\n      schema,\n      batch_size=_TRAIN_BATCH_SIZE)\n  eval_dataset = _input_fn(\n      fn_args.eval_files,\n      fn_args.data_accessor,\n      schema,\n      batch_size=_EVAL_BATCH_SIZE)\n\n  model = _build_keras_model()\n  model.fit(\n      train_dataset,\n      steps_per_epoch=fn_args.train_steps,\n      validation_data=eval_dataset,\n      validation_steps=fn_args.eval_steps)\n\n  # The result of the training should be saved in `fn_args.serving_model_dir`\n  # directory.\n  model.save(fn_args.serving_model_dir, save_format=\'tf\')')


# ### Write a pipeline definition
# 
# Next, you will define a function to create a TFX pipeline. A [`Pipeline`](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/dsl/Pipeline) object represents a TFX pipeline which can be run using one of pipeline orchestration systems that TFX supports.

# In[ ]:


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     metadata_path: str) -> tfx.dsl.Pipeline:
  """Creates a three component penguin pipeline with TFX."""
  # Brings data into the pipeline.
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)

  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=example_gen.outputs['examples'],
      train_args=tfx.proto.TrainArgs(num_steps=100),
      eval_args=tfx.proto.EvalArgs(num_steps=5))

  # Pushes the model to a filesystem destination.
  pusher = tfx.components.Pusher(
      model=trainer.outputs['model'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  # Following three components will be included in the pipeline.
  components = [
      example_gen,
      trainer,
      pusher,
  ]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      components=components)


# ## Run the pipeline
# 
# TFX supports multiple orchestrators to run pipelines.
# In this tutorial we will use `LocalDagRunner` which is included in the TFX
# Python package and runs pipelines on local environment.
# We often call TFX pipelines "DAGs" which stands for directed acyclic graph.
# 
# `LocalDagRunner` provides fast iterations for developemnt and debugging.
# TFX also supports other orchestrators including Kubeflow Pipelines and Apache
# Airflow which are suitable for production use cases.
# 
# See
# [TFX on Cloud AI Platform Pipelines](https://www.tensorflow.org/tfx/tutorials/tfx/cloud-ai-platform-pipelines)
# or
# [TFX Airflow Tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/airflow_workshop)
# to learn more about other orchestration systems.
# 
# The code below creates a `LocalDagRunner` and passes a `Pipeline` object created from the
# function you defined earlier. The pipeline runs directly and you can see logs for the progress of the pipeline including ML model training.

# In[ ]:


tfx.orchestration.LocalDagRunner().run(
  _create_pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      data_root=DATA_ROOT,
      module_file=_trainer_module_file,
      serving_model_dir=SERVING_MODEL_DIR,
      metadata_path=METADATA_PATH))


# You should see "INFO:absl:Component Pusher is finished." at the end of the
# logs if the pipeline finished successfully. Because `Pusher` component is the
# last component of the pipeline. You can also see that the `metadata` (metadata store), `pipelines` (pipeline root), and `serving_model` directories have been populated with the metadata, artifacts, and models that the pipeline generates.
# 
# Now that you've ran this simple pipeline, you will see in the next sections how you can modify it to use custom components.

# ## Building Custom Components
# 
# Let's say you want to modify the pipeline above to filter the data first before running the trainer. Without using a TFX component, you would run something like the code below. This will just get the rows where the `culmen_length_mm` feature is greater than `0.3`.

# In[ ]:


# search directory for one or more CSVs
files = glob.glob(f'{DATA_ROOT}/*.csv')

# filter the dataset
for file in files:
  df = pd.read_csv(file, index_col=False)
  filtered_df = df[df['culmen_length_mm'] > 0.3].reset_index(drop=True)

# print latest modified file
filtered_df


# You can save the dataset above to a different directory and point the TFX pipeline to it. That definitely works but you can include this code in a custom component as well so it's part of the TFX pipeline.
# 
# As mentioned in the lectures, a TFX component has a driver, executor, and publisher. The driver and publisher interacts with the metadata store while the executor runs the actual processing. More often than not, you only need to modify the executor and that's what you'll be doing in the next sections.

# ## Custom components through Python functions
# 
# TFX provides a way to define an executor by just using a `component` decorator on your function. If you've done the Kubeflow Pipelines ungraded lab earlier this week, this will look very familiar. It uses the same concepts such as defining the inputs and outputs using type annotations then using those parameters in the function body.
# 
# The component below basically uses the same filtering code you saw earlier but wraps it in a Python component. It defines two input parameters and outputs a dictionary with a String artifact (see function docstring for description). Because this component uses the same publisher as a standard TFX component, the output artifact will be saved in the metadata store as you'll see later.

# In[ ]:


#import artifact type that you will use
from tfx.types.standard_artifacts import String

# import the decorator
from tfx.dsl.component.experimental.decorators import component

# import type annotations
from tfx.dsl.component.experimental.annotations import InputArtifact, OutputArtifact, Parameter, OutputDict

@component
def CustomFilterComponent(input_base: Parameter[str], 
                    output_base: Parameter[str],
                    ) -> OutputDict(output_path=str):
  '''
  Args:
    input_base - location of the raw CSV
    output_base - location where you want to save the filtered CSV
  
  Returns:
    OutputDict:
      output_path - String artifact that just holds the `output_base` value
  '''
  import pandas as pd
  import glob
  import os

  # create the output base if it does not exist yet
  if not os.path.exists(output_base):
      os.mkdir(output_base)
  
  # search for CSVs in the input base
  files = glob.glob(f'{input_base}/*.csv')

  # loop through CSVs
  for file in files:

    # read the CSV
    df = pd.read_csv(file, index_col=False)

    # filter the data
    filtered_df = df[df['culmen_length_mm'] > 0.3].reset_index(drop=True)

    # compose output filename
    filename = os.path.basename(file).replace('.csv','')
    filtered_filename = f'{filename}_filtered.csv'
  
    # save filtered CSV to output base
    filtered_df.to_csv(f'{output_base}/{filtered_filename}', index=False)

  # define the output artifact
  return {'output_path': output_base}


# You can now run your newly built component. You will just run a single node pipeline to prove that it works.

# In[ ]:


# define a filter task
filter_task = CustomFilterComponent(input_base=DATA_ROOT, 
                                        output_base=f'{DATA_ROOT}_filtered')

# include the task
components = [filter_task]

# define a pipeline with only the single component
pipeline = tfx.dsl.Pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(METADATA_PATH),
      components=components)

# run the pipeline
tfx.orchestration.LocalDagRunner().run(pipeline)


# You should now see the filtered CSV in the `data_filtered` folder in the file explorer. As expected, you can also see that it has less lines than the original because of the filtering.

# In[ ]:


# number of rows in original csv
get_ipython().system('cat data/data.csv | wc -l')

# number of rows in filtered csv
get_ipython().system('cat data_filtered/data_filtered.csv | wc -l')


# If you navigate to `pipelines/penguin-simple/CustomFilterComponent/output_path/`, you will also see a directory with the run id of the pipeline (e.g. `4`). If you click on the `value` file beneath it, you'll see the string value you saved which is just the `output_base` (i.e. `data_filtered`). You will want to feed this value to the `CsvExampleGen` component and that's what you'll do next.

# ### Building a custom component using standard components
# 
# If we try to use our new component with CsvExampleGen, **you will encounter an error** as shown below:

# In[ ]:


# Define filter task
filter_task = CustomFilterComponent(input_base=DATA_ROOT, 
                                        output_base=f'{DATA_ROOT}_filtered')

# Try using the custom component with CsvExampleGen. This code will expectedly throw an error.
try:
  example_gen = tfx.components.CsvExampleGen(input_base=filter_task.outputs['output_path'])

except Exception as e:
  print("Error thrown as expected!")
  print(e)


# As you can see, the output of our custom component cannot be accepted into `CsvExampleGen` because it is configured to accept a primitive string. TFX components generate and consume [`Channel`](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/dsl/Channel) objects and that's what our custom component outputs. For that, we need to build a custom data ingestion component that reuses the `CsvExampleGen` code but accepts a `Channel`. You will do that in the following sections.

# ### Standard ExampleGen code
# 
# TFX is open source so the code for standard components can easily be found the public [repo](https://github.com/tensorflow/tfx/tree/89d3cb6c59acd0d487916bff703711815f1506b5/tfx). We placed links in the following sections of the actual files that you'll be modifying/overriding in case you want to compare what was changed for your custom ExampleGen. 
# 
# The class heirarchy for these components is pretty deep but in summary, you will only need to modify three:
# 
# * Component Spec - this describes the inputs, outputs, and parameters passed on to the component
# * Executor - code for processing the inputs, outputs, and parameters
# * Component Class - puts everything together so your code can be run by the TFX runner.

# ### Modify the Component Spec
# 
# First, you will need to modify the Component Spec. This file describes the parameters, inputs, and outputs of the standard components. Parameters are values supplied at runtime while inputs and outputs are values read from the metadata store. The original for ExampleGen is found [here](https://github.com/tensorflow/tfx/blob/89d3cb6c59acd0d487916bff703711815f1506b5/tfx/types/standard_component_specs.py#L187).
# 
# You will need to implement the same for our custom ExampleGen but it should accept a `Channel` parameter The revised version is shown below. Notice that we revised the `INPUTS` dictionary to have a `ChannelParameter` whereas in the original, all are just in the `PARAMETERS` dictionary.

# In[ ]:


from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types import standard_artifacts
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2


# Key for example_gen input that we want to use
INPUT_BASE_KEY = 'input_base'

# Other keys
INPUT_CONFIG_KEY = 'input_config'
OUTPUT_CONFIG_KEY = 'output_config'
OUTPUT_DATA_FORMAT_KEY = 'output_data_format'
RANGE_CONFIG_KEY = 'range_config'
CUSTOM_CONFIG_KEY = 'custom_config'
EXAMPLES_KEY = 'examples'

class MyCustomExampleGenSpec(ComponentSpec):
  """File-based ExampleGen component spec."""
  
  PARAMETERS = {
      INPUT_CONFIG_KEY:
          ExecutionParameter(type=example_gen_pb2.Input),
      OUTPUT_CONFIG_KEY:
          ExecutionParameter(type=example_gen_pb2.Output),
      OUTPUT_DATA_FORMAT_KEY:
          ExecutionParameter(type=int),
      CUSTOM_CONFIG_KEY:
          ExecutionParameter(type=example_gen_pb2.CustomConfig, optional=True),
      RANGE_CONFIG_KEY:
          ExecutionParameter(type=range_config_pb2.RangeConfig, optional=True),
  }

  # Now accepts a channel
  INPUTS = {
      INPUT_BASE_KEY:
          ChannelParameter(type=standard_artifacts.String),
  }
  
  OUTPUTS = {
      EXAMPLES_KEY: ChannelParameter(type=standard_artifacts.Examples),
  }


# ### Customize the Executor
# 
# With that, you should now modify the executor code to take note of this change of input types. Instead of looking at just the parameters, it should also look into Channel inputs passed onto the component.
# 
# Executor classes are executed by TFX starting with the `Do` function so you will need to modify that. The original Executor for `CsvExampleGen` can be found [here](https://github.com/tensorflow/tfx/blob/89d3cb6c59acd0d487916bff703711815f1506b5/tfx/components/example_gen/csv_example_gen/executor.py) and it inherits the base class [here](https://github.com/tensorflow/tfx/blob/89d3cb6c59acd0d487916bff703711815f1506b5/tfx/components/example_gen/base_example_gen_executor.py#L132). The base class includes the `Do()` function and that's what you'll be overriding in your new custom executor below. 
# 
# Basically, you're using all the functions defined in the standard component but you're modifying it so it can find the `input_base` value from the Channel inputs.
# 
# *Take note that this tutorial prioritizes code brevity. In your projects, you may take a different approach such as modifying the [`_CsvToExample`](https://github.com/tensorflow/tfx/blob/89d3cb6c59acd0d487916bff703711815f1506b5/tfx/components/example_gen/csv_example_gen/executor.py#L124) code to look for the string value in the `input_dict` instead of `exec_properties`. Doing that here will result in very long code blocks so the shorter approach is taken.*

# In[ ]:


from typing import Any, Dict, Iterable, List, Text, Union

from tfx.components.example_gen.csv_example_gen.executor import Executor as CsvExampleGenExecutor
from tfx.types import standard_component_specs
from tfx.types import artifact_utils
from tfx import types

class MyCustomExecutor(CsvExampleGenExecutor):
  """Generic TFX CSV example gen executor."""

  def Do(
      self,
      input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Any],
  ) -> None:
    """Take input data source and generates serialized data splits.
    The output is intended to be serialized tf.train.Examples or
    tf.train.SequenceExamples protocol buffer in gzipped TFRecord format,
    but subclasses can choose to override to write to any serialized records
    payload into gzipped TFRecord as specified, so long as downstream
    component can consume it. The format of payload is added to
    `payload_format` custom property of the output Example artifact.
    Args:
      input_dict: Input dict from input key to a list of Artifacts. Depends on
        detailed example gen implementation.
        - input_base: an external directory containing the data files.
      output_dict: Output dict from output key to a list of Artifacts.
        - examples: splits of serialized records.
      exec_properties: A dict of execution properties. Depends on detailed
        example gen implementation.
        - input_config: JSON string of example_gen_pb2.Input instance,
          providing input configuration.
        - output_config: JSON string of example_gen_pb2.Output instance,
          providing output configuration.
        - output_data_format: Payload format of generated data in output
          artifact, one of example_gen_pb2.PayloadFormat enum.
    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # Get the artifact from the Channel input
    filter_component_artifact = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.INPUT_BASE_KEY])
    
    # Put the input string value into the exec_properties fictionary
    exec_properties[standard_component_specs.INPUT_BASE_KEY] = filter_component_artifact.value
    
    # execute superclass
    super(MyCustomExecutor, self).Do(input_dict=input_dict, output_dict=output_dict, exec_properties=exec_properties)


# ### Define the Component class
# 
# Lastly, you will need to put everything together in a class. This will be the one you instantiate so you can run the component later. For comparison, the original `CsvExampleGen` component class is found [here](https://github.com/tensorflow/tfx/blob/89d3cb6c59acd0d487916bff703711815f1506b5/tfx/components/example_gen/csv_example_gen/component.py) and it inherits the `FileBasedExampleGen` from [here](https://github.com/tensorflow/tfx/blob/89d3cb6c59acd0d487916bff703711815f1506b5/tfx/components/example_gen/component.py#L115). The revised version is shown below:

# In[ ]:


from typing import Any, Dict, Optional, Text, Union

from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import executor_spec

from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2

from tfx import types
from tfx.components.example_gen import utils

class MyCustomExampleGen(base_beam_component.BaseBeamComponent):

  # Define the Spec class and executor spec using the functions and
  # classes you defined earlier.
  SPEC_CLASS = MyCustomExampleGenSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(MyCustomExecutor)

  # Define init function. Notice that `input_base` now accepts a Channel.
  def __init__(self,
               input_base: types.Channel = None,
               input_config: Optional[Union[example_gen_pb2.Input,
                                            Dict[Text, Any]]] = None,
               output_config: Optional[Union[example_gen_pb2.Output,
                                             Dict[Text, Any]]] = None,
               range_config: Optional[Union[range_config_pb2.RangeConfig,
                                            Dict[Text, Any]]] = None,
               output_data_format: Optional[int] = example_gen_pb2.FORMAT_TF_EXAMPLE):
    """Customized ExampleGen component.
    Args:
      input_base: an external directory containing the CSV files. Accepts a Channel
        from a previous TFX component.
      input_config: An example_gen_pb2.Input instance, providing input
        configuration. If unset, the files under input_base will be treated as a
        single split. If any field is provided as a RuntimeParameter,
        input_config should be constructed as a dict with the same field names
        as Input proto message.
      output_config: An example_gen_pb2.Output instance, providing output
        configuration. If unset, default splits will be 'train' and 'eval' with
        size 2:1. If any field is provided as a RuntimeParameter, output_config
        should be constructed as a dict with the same field names as Output
        proto message.
      range_config: An optional range_config_pb2.RangeConfig instance,
        specifying the range of span values to consider. If unset, driver will
        default to searching for latest span with no restrictions.
    """
    
    # Configure inputs and outputs.
    input_config = input_config or utils.make_default_input_config()
    output_config = output_config or utils.make_default_output_config(
        input_config)
    
    # Define output type.
    example_artifacts = types.Channel(type=standard_artifacts.Examples)
    
    # Pass input arguments to your custom ExampleGen spec.
    spec = MyCustomExampleGenSpec(
        input_base=input_base,
        input_config=input_config,
        output_config=output_config,
        range_config=range_config,
        output_data_format=output_data_format,
        examples=example_artifacts)
    
    # This will check if the values passed are the correct type else
    # it will throw the error you saw earlier.
    super(MyCustomExampleGen, self).__init__(
        spec=spec)
  


# You can now use the custom component (`MyCustomExampleGen`) in your code as shown below. It will no longer get an error because you reconfigured the `input_base` to accept a channel.

# In[ ]:


# Filter the dataset
filter_task = CustomFilterComponent(input_base=DATA_ROOT, 
                                        output_base=f'{DATA_ROOT}_filtered')

# Use the output of filter_task to know the input_base for this custom ExampleGen
custom_example_gen_task = MyCustomExampleGen(input_base=filter_task.outputs['output_path'])

# Define components to include
components = [filter_task,
              custom_example_gen_task]

# Create the pipeline
pipeline = tfx.dsl.Pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(METADATA_PATH),
      components=components)

# Run the pipeline
tfx.orchestration.LocalDagRunner().run(pipeline)


# As a sanity check, you can compute the number of examples for both the training and eval splits. It should equal the number of examples found in your filtered CSV. You can use the code below by replacing the `EXECUTION_ID` with the ID shown in your latest run. You can see it in the last three lines of the output cell above. For example:
# 
# ```
# )]}) for execution 16  # ---> **16 is the EXECUTION ID**
# INFO:absl:MetadataStore with DB connection initialized
# INFO:absl:Component MyCustomExampleGen is finished.
# ```

# In[ ]:


EXECUTION_ID = 6 # PLACE THE EXECUTION ID HERE

# Create a `TFRecordDataset` to read these files
train_dataset = tf.data.TFRecordDataset(f'{PIPELINE_ROOT}/MyCustomExampleGen/examples/{EXECUTION_ID}/Split-train/data_tfrecord-00000-of-00001.gz', compression_type="GZIP")
eval_dataset = tf.data.TFRecordDataset(f'{PIPELINE_ROOT}/MyCustomExampleGen/examples/{EXECUTION_ID}/Split-eval/data_tfrecord-00000-of-00001.gz', compression_type="GZIP")

# Get number of records for each dataset (only use for small datasets to avoid memory issues)
num_train_data = len(list(train_dataset))
num_eval_data = len(list(eval_dataset))

# Get the total
total_examples = num_train_data + num_eval_data

print(f'total number of examples: {total_examples}')


# ## Wrap Up
# 
# In this lab, you were able to use custom components to create a pipeline. This shows that you can go outside the standard TFX components if your project calls for it. To know more about custom components, you can read more [here](https://www.tensorflow.org/tfx/guide/understanding_custom_components) and see the examples [here](https://github.com/tensorflow/tfx/tree/2e41786328f5b69720e90ec4d9ecae500f5c157a/tfx/examples/custom_components).

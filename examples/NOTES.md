# Notes

## TF Transform

challeng is to keep Transforms in sycn between train/serve. If they vary, then this is called *training skew*

**Analyzers** - do a full Dataset pass to calculate a statistic

**Transformers (mappers)** - are pure functions that Transform a Tensor to a Tensor

## Examples

### sentiment_example.py

`read_and_shuffle_data`

- reads in data and shuffles and writes to 2 separate dirs - train/test

`transform_data`

- reads in TFRecords from previous step
- does TF Transform
- writes transformed records to separate files
- writes `tf_transform` artifacts - which work with the transformed data artifacts - that can now be used for trainging

  - `transform_fn`
  - `transformed_metadata`


`train_and_evaluate`

- train and evaluates the Model
- uses the transformed data to to so
- Model will generate `exported_model` artifacts to be used for Serving the Model

NOTE: there is not "Serving a Model" example in `sentiment_example.py`

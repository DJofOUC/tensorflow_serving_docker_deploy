import parser
import shutil

import tensorflow as tf
from wide_deep1 import build_model_columns

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]


def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous variable columns
  age = tf.feature_column.numeric_column('age')
  education_num = tf.feature_column.numeric_column('education_num')
  capital_gain = tf.feature_column.numeric_column('capital_gain')
  capital_loss = tf.feature_column.numeric_column('capital_loss')
  hours_per_week = tf.feature_column.numeric_column('hours_per_week')

  education = tf.feature_column.categorical_column_with_vocabulary_list(
      'education', [
          'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
          'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
          '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
      'marital_status', [
          'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

  relationship = tf.feature_column.categorical_column_with_vocabulary_list(
      'relationship', [
          'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
          'Other-relative'])

  workclass = tf.feature_column.categorical_column_with_vocabulary_list(
      'workclass', [
          'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
          'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

  # To show an example of hashing:
  occupation = tf.feature_column.categorical_column_with_hash_bucket(
      'occupation', hash_bucket_size=1000)

  # Transformations.
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  base_columns = [
      education, marital_status, relationship, workclass, occupation,
      age_buckets,
  ]

  crossed_columns = [
      tf.feature_column.crossed_column(
          ['education', 'occupation'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          [age_buckets, 'education', 'occupation'],
          hash_bucket_size=1000),
  ]

  wide_columns = base_columns + crossed_columns

  deep_columns = [
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
      tf.feature_column.indicator_column(workclass),
      tf.feature_column.indicator_column(education),
      tf.feature_column.indicator_column(marital_status),
      tf.feature_column.indicator_column(relationship),
      # To show an example of embedding
      tf.feature_column.embedding_column(occupation, dimension=8),
  ]

  return wide_columns, deep_columns

def input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), ('no file named : ' + str(data_file))

    def process_list_column(list_column):
        sparse_strings = tf.string_split(list_column, delimiter="|")
        return sparse_strings.values

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        features['workclass'] = process_list_column([features['workclass']])
        labels = tf.equal(features.pop('income_bracket'), '>50K')
        labels = tf.reshape(labels, [-1])
        return features, labels

    dataset = tf.data.TextLineDataset(data_file)
    if shuffle: dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(parse_csv)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset

def build_estimator(model_dir):

    wide_columns,deep_columns=build_model_columns()
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        n_classes=2,
        dnn_dropout=0.5,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100,100,200],
        config=run_config)

def main():

    shutil.rmtree('model', ignore_errors=True)
    model = build_estimator('model')

    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(10// 2):
        model.train(input_fn=lambda: input_fn('adult.data', 2, True, 10))
        results = model.evaluate(input_fn=lambda: input_fn('adult.test', 1, False, 100))

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * 2)
        print('-' * 20)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))

    wide_columns,deep_columns= build_model_columns()
    fs=wide_columns+deep_columns
    fs = tf.feature_column.make_parse_example_spec(fs)
    print(fs)
    serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(fs)
    print('>>>\n',serving_fn)
    export_dir = model.export_savedmodel('export',serving_fn,as_text=True)
    print(export_dir)


def predict():
    model = build_estimator('model')
    predictions = model.predict(input_fn=lambda: input_fn('adult.test', 1, False, 40))
    file=open('predict_result.txt','w')
'''
def get_input(a_list):
    def _float_feature(value):
        if value=='':
            value=0.0
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))

    def _byte_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
     age,workclass,fnlwgt,education,education_num,marital_status,occupation,
     relationship,race,gender,capital_gain,capital_loss,hours_per_week,native_country,income_bracket
     feature_dict={
        'age':-_float_feature(age),
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
    }

def predict_model():
    predict_fn = tf.contrib.predictor.from_saved_model('export/1533442948')

    return 0
'''
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    #FLAGS, unparsed = parser.parse_known_args()
    main()
    #tf.app.run(main=main, argv='1')

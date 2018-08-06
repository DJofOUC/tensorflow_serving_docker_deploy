# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations

#
def get_input(a_list):

	def _float_feature(value):
		if value=='':
			value=0.0
		return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))

	def _byte_feature(value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	'''
	age,workclass,fnlwgt,education,education_num,marital_status,occupation,
	relationship,race,gender,capital_gain,capital_loss,hours_per_week,
	native_country,income_bracket=a_list.strip('\n').strip('.').split(',')
	'''
	feature_dict={
		'age':_float_feature(a_list[0]),
		'workclass':_byte_feature(a_list[1].encode()),
		'education':_byte_feature(a_list[3].encode()),
		'education_num':_float_feature(a_list[4]),
		'marital_status':_byte_feature(a_list[5].encode()),
		'occupation':_byte_feature(a_list[6].encode()),
		'relationship':_byte_feature(a_list[7].encode()),
		'capital_gain':_float_feature(a_list[10]),
		'capital_loss':_float_feature(a_list[11]),
		'hours_per_week':_float_feature(a_list[12]),
	}
	model_input=tf.train.Example(features=tf.train.Features(feature=feature_dict))
	return model_input

def main():
	channel = implementations.insecure_channel('10.211.44.8', 8500)#the ip and port of your server host
	stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

	# the test samples
	examples = []
	f=open('adult.test','r')
	for line in f:
		line=line.strip('\n').strip('.').split(',')
		example=get_input(line)
		examples.append(example)

	request = classification_pb2.ClassificationRequest()
	request.model_spec.name = 'dnn'#your model_name which you set in docker  container
	request.input.example_list.examples.extend(examples)

	response = stub.Classify(request, 20.0)

	for index in range(len(examples)):
		print(index)
		max_class = max(response.result.classifications[index].classes, key=lambda c: c.score)
		re=response.result.classifications[index]
		print(max_class.label,max_class.score)# the prediction class and probability
if __name__=='__main__':
	main()

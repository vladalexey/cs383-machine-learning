import numpy as np
import os
import sys
import pandas as pd
from math import sqrt

np.random.seed(5983)
DIR = './insurance.csv'

def split_dataset(data):
	shuffled = data.sample(frac=1, random_state=2452)

	test = shuffled.iloc[:len(data) // 3, :].reset_index(drop=True)
	train = shuffled.iloc[len(data) // 3:, :].reset_index(drop=True)
	print(test)
	return train, test

def get_disc_vals(df, col_name):

	values = dict()
	cnt = 0
	for val in df[col_name]:
		if val not in values:
			values[val] = cnt
			cnt += 1

	return values

def hot_encode(df, col_name):

	vals = get_disc_vals(df, col_name)

	for val in vals:
		df[col_name + '_' + str(int(val))] = 1
	
	for row, val in enumerate(df[col_name]):
		df.at[row, col_name + '_' + str(int(val))] = 2

	df = df.drop(columns=[col_name])

	return df

def add_noise(df, col_name):

	for row, val in enumerate(df[col_name]):
		df.at[row, col_name + '_' + str(int(val))] = 2

def transform(df, col_name):

	vals = get_disc_vals(df, col_name)

	for idx, row in enumerate(df[col_name]):
		df.at[idx, col_name] = float(vals[row])
	
	return df

def system_ONE(data):

	x = data.iloc[:, :-1]
	y = data.iloc[:, -1]
	
	for feature in ('region', 'sex', 'smoker'):
		x = transform(x, feature)

	X = np.array(x.values, dtype=float)
	w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

	preds = X.dot(w)
	rmse = sqrt(np.square(np.subtract(y,preds)).mean())
	print("SYSTEM 1:\nWeight: {}\nRMSE: {}".format(w, rmse))

def system_TWO(data):

	x = data.iloc[:, :-1]
	y = data.iloc[:, -1]
	
	for feature in ('region', 'sex', 'smoker'):
		x = transform(x, feature)
	x['bias'] = 1

	X = np.array(x.values, dtype=float)
	w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
	preds = X.dot(w)
	rmse = sqrt(np.square(np.subtract(y,preds)).mean())
	print("SYSTEM 2:\nWeight: {}\nRMSE: {}".format(w, rmse))


def system_THREE(data):
	x = data.iloc[:, :-1]
	y = data.iloc[:, -1]

	for feature in ('region', 'sex', 'smoker'):
		x = transform(x, feature)

	x = hot_encode(x, 'region')

	X = np.array(x.values, dtype=float)
	noise = np.random.normal(0, 0.5, X[:, -4:].shape)
	X[:, -4:] = X[:, -4:] + noise

	w = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
	preds = X.dot(w)
	rmse = sqrt(np.square(np.subtract(y,preds)).mean())
	print("SYSTEM 3:\nWeight: {}\nRMSE: {}".format(w, rmse))

def system_FOUR(data):
	x = data.iloc[:, :-1]
	y = data.iloc[:, -1]

	for feature in ('region', 'sex', 'smoker'):
		x = transform(x, feature)
	x['bias'] = 1
	x = hot_encode(x, 'region')
	
	X = np.array(x.values, dtype=float)
	noise = np.random.normal(0, 0.5, X[:, -4:].shape)
	X[:, -4:] = X[:, -4:] + noise

	w = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
	preds = X.dot(w)
	rmse = sqrt(np.square(np.subtract(y,preds)).mean())
	print("SYSTEM 4:\nWeight: {}\nRMSE: {}".format(w, rmse))

if __name__ == '__main__':

	data = pd.read_csv(DIR)
	train, test = split_dataset(data)

	print('================================================================')
	print('----------------------Training----------------------')
	system_ONE(train)
	print('----------------------Testing----------------------')
	system_ONE(test)
	print('================================================================')
	print('----------------------Training----------------------')
	system_TWO(train)
	print('----------------------Testing----------------------')
	system_TWO(test)
	print('================================================================')
	print('----------------------Training----------------------')
	system_THREE(train)
	print('----------------------Testing----------------------')
	system_THREE(test)
	print('================================================================')
	print('----------------------Training----------------------')
	system_FOUR(train)
	print('----------------------Testing----------------------')
	system_FOUR(test)
	print('================================================================')
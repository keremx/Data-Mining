import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split

def get_data(file_path):
	"""
	Takes an input of a filepath which has the training data
	Returns 2 dictionaries (data_dict and opt_dict)
	data_dict: Contains 3 keys (users, items, ratings) each containing a list
	opt_dict: Contains a single key (users_items) containing a list of lists with each list having user and it's rated item
	"""
	data_dict = {
		"users": [],
		"items" : [],
		"ratings": []
	}
	opt_dict = {
		"users_items": []
	}
	file = open(file_path,'r')
	for line in file:
		line = (line.replace("\n","")).split(" ")
		data_dict["users"].append(int(line[0]))
		data_dict["items"].append(int(line[1]))
		data_dict["ratings"].append(int(line[2]))
		opt_dict["users_items"].append([int(line[0]), int(line[1])])
	return data_dict, opt_dict

def split_data(data, test_size):
	"""
	Takes data and test_size as input.
	Returns trainset and testset which are surprise objects.
	"""
	trainset, testset = train_test_split(data, test_size=test_size)
	return trainset, testset

def train(trainset):
	"""
	Item-based cosine KNN: For an item I, with a set of similar items determined based on rating vectors consisting of received user ratings, 
	the rating by a user U, who hasn’t rated it, is found by picking out N items from the similarity list 
	(calculated using cosine angle - if angel between two vectors is less it means that they are similar) 
	that have been rated by U and calculating the rating based on these N ratings.
	
	Function takes a trainset as input argument and fits the data on KNNWithMeans algo having "cosine" and "item-based" collaborative filtering
	Returns the fitted model
	"""

	my_sim_option = {'name':'cosine', 'user_based':False}
	model = KNNWithMeans(sim_option = my_sim_option)
	model = model.fit(trainset)	
	return model

def test(model, testset):
	"""
	Function takes fitted model and testset as input arguments and calculates root mean square error
	Returns the calculated root mean square error
	"""
	predictions = model.test(testset)
	rmse = accuracy.rmse(predictions)
	return rmse

def save_output(output_file_path, model, data_dict, opt_dict):
	"""
	Function takes as input an output file path with the name of the file, fitted model, 
	data_dict and opt_dict (data_dict and opt_dict retrieved from get_data function)
	Predicts results for users that have not rated a specific item and the rest of the actual results to the output file path 
	"""
	output_file = open(output_file_path, 'w')
	items_list = list(range(min(data_dict['items']), max(data_dict['items'])+1))
	users_list = list(range(min(data_dict['users']), max(data_dict['users'])+1))
	ratings = None 
	for i in range(0, len(users_list)):
		for j in range(0, len(items_list)):
			users_items = [users_list[i], items_list[j]]
			if users_items not in opt_dict['users_items']:
				prediction = model.predict(users_list[i], items_list[j])
				ratings = int(prediction.est)
			elif users_items in opt_dict['users_items']:
				ratings = data_dict['ratings'][opt_dict['users_items'].index(users_items)]
			print("{} {} {}".format(users_list[i], items_list[j], ratings))
			output_file.write("{} {} {}\n".format(users_list[i], items_list[j], ratings))


if __name__ == "__main__":
	"""
	Main function which runs as the program starts and calls all the above defined functions in that exact order.
	"""
	data_dict, opt_dict = get_data('train.txt')
	df = pd.DataFrame(data_dict)
	reader = Reader(rating_scale=(min(data_dict['ratings']), max(data_dict['ratings'])))

	data = Dataset.load_from_df(df[["users", "items", "ratings"]], reader)

	trainset, testset = split_data(data, 0.2)
	trained_model = train(trainset) 
	rmse = test(trained_model, testset)

	save_output('output_matrix.txt', trained_model, data_dict, opt_dict)

	trainsetfull = data.build_full_trainset()
	print('\n\nNumber of users in training data: {}'.format(trainsetfull.n_users))
	print('Number of items ranked in training data: {}'.format(trainsetfull.n_items))
	print('\nTotal number of users: {}'.format(max(data_dict['users'])))
	print('Total number of items: {}'.format(max(data_dict['items'])))
	print("\nRMSE: {}".format(rmse))


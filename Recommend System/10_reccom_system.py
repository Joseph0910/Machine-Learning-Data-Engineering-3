class RecommEngine():

	def __init__(self,train_data, name, user_id, item_id, target, users_to_recommend, 
				n_rec, n_display,customer_id,df_output):
		self.train_data = train_data
		self.name  = name
		self.user_id = user_id
		self.item_id = item_id
		self.target = target
		self.users_to_recommend = users_to_recommend
		self.n_rec = n_rec
		self.n_display = n_display
		self.customer_id = customer_id
		self.df_output = df_output
		
	# constant variables to define field names include:
#user_id = 'customerId'
#item_id = 'productId'
#n_rec = 10 # number of items to recommend
#n_display = 30 # to display the first few rows in an output dataset

	def model_construct():
		users_to_recommend = list(customers[user_id])
		%load_ext autoreload
		%autoreload 2

		sys.path.append("..")

	    if name == 'popularity': # baseline model # 
	        model = tc.popularity_recommender.create(train_data, 
	                                                    user_id=user_id, 
	                                                    item_id=item_id, 
	                                                    target=target)
	    elif name == 'cosine':
	        model = tc.item_similarity_recommender.create(train_data, 
	                                                    user_id=user_id, 
	                                                    item_id=item_id, 
	                                                    target=target, 
	                                                    similarity_type='cosine')
		elif name == 'pearson':
		        model = tc.item_similarity_recommender.create(train_data, 
		                                                    user_id=user_id, 
		                                                    item_id=item_id, 
		                                                    target=target, 
		                                                    similarity_type='pearson')
		        
		recom = model.recommend(users=users_to_recommend, k=n_rec)
		recom.print_rows(n_display)
		
		return model

	def model_evaluation(self):

		models_w_counts = [popularity_model, cos, pear]
		models_w_dummy = [pop_dummy, cos_dummy, pear_dummy]
		models_w_norm = [pop_norm, cos_norm, pear_norm]
		names_w_counts = ['Popularity Model on Purchase Counts', 'Cosine Similarity on Purchase Counts', 'Pearson Similarity on Purchase Counts']
		names_w_dummy = ['Popularity Model on Purchase Dummy', 'Cosine Similarity on Purchase Dummy', 'Pearson Similarity on Purchase Dummy']
		names_w_norm = ['Popularity Model on Scaled Purchase Counts', 'Cosine Similarity on Scaled Purchase Counts', 'Pearson Similarity on Scaled Purchase Counts']

		eval_counts = tc.recommender.util.compare_models(test_data, models_w_counts, model_names=names_w_counts)
		eval_dummy = tc.recommender.util.compare_models(test_data_dummy, models_w_dummy, model_names=names_w_dummy)
		eval_norm = tc.recommender.util.compare_models(test_data_norm, models_w_norm, model_names=names_w_norm)


	def recommendation_function(self):
	    if customer_id not in df_output.index:
	        print('Customer not found.')
	        return customer_id
	    return df_output.loc[customer_id]


# using purchase count #
name = 'popularity'
target = 'purchase_count'
popularity = model_construct(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'cosine'
target = 'purchase_count'
cos = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'pearson'
target = 'purchase_count'
pear = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# Using purchase dummy #
name = 'popularity'
target = 'purchase_dummy'
pop_dummy = model_construct(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'cosine'
target = 'purchase_dummy'
cos_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'pearson'
target = 'purchase_dummy'
pear_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# using scaled purchase count #

name = 'popularity'
target = 'scaled_purchase_freq'
pop_norm = model_construct(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


name = 'cosine' 
target = 'scaled_purchase_freq' 
cos_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

name = 'pearson'
target = 'scaled_purchase_freq'
pear_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# Final Model #
final_model = tc.item_similarity_recommender.create(tc.SFrame(data_norm), 
                                            user_id=user_id, 
                                            item_id=item_id, 
                                            target='purchase_dummy', similarity_type='cosine')
recom = final_model.recommend(users=users_to_recommend, k=n_rec)
recom.print_rows(n_display)

import pandas as pd
import numpy as np

class Preprocess:
	def __init__(self, item, user, behavior):
		self.dfitem = pd.read_csv(item)
		self.dfuser = pd.read_csv(user)
		self.dfbehavior = pd.read_csv(behavior)

	def init_process(self):
		dfuser_column = ['user_id', 'gender', 'age', 'power']
		dfitem_column = ['item_id', 'cat_id', 'shop_id', 'brand_id']
		dfbehavior_column = ['user_id', 'item_id', 'type', 'time']

		self.dfitem.columns = dfitem_column
		self.dfuser.columns = dfuser_column
		self.dfbehavior.columns = dfbehavior_column

		df = self.dfbehavior.merge(self.dfitem, on = 'item_id').merge(self.dfuser, on = 'user_id')

		df = df.sort_values(['user_id', 'time'], ascending=[True, True])
		df['type'] = df['type'].replace(['buy', 'pv', 'cart', 'fav'], ['b','p','c','f'])
		
		return df

import matplotlib.pyplot as plt
from gensim.models.doc2vec import TaggedDocument
from collections import Counter
from nltk import ngrams
import numpy as np



class SessionClassifier:

	def __init__(self):
		self.threshold = 10000 #人工设定，真实数据可直接用是否为新session代替

	def set_time_threshold(self, threshold):
		self.threshold = threshold

	def build_time_interval(self, userframe, timeframe):
		result = [self.threshold]
		for i in range(1, len(timeframe)):
			prev_id = userframe[i-1]
			curr_id = userframe[i]
			if prev_id == curr_id:
				result.append(timeframe[i]-timeframe[i-1])
			else:
				result.append(self.threshold)
		return result

	def build_time_threshold(self, interval):
		result = [1]
		for i in range(1, len(interval)):
			if interval[i] >= self.threshold:
				result.append(1)
			else:
				result.append(0)
		return result

	def stamp_to_signal(self, userframe, timeframe):
		return self.build_time_threshold(self.build_time_interval(userframe, timeframe))

class PathEncoder:

	def __init__(self):
		self.id = 0
		self.type = 2
		self.cat = 5
		self.session = 10

	def set_cat(self, cat):
		self.cat = cat

	def get_abbr_path(self, lst):
		dic = dict()
		result = []
		actions = ''
		actions_sum = ''

		prev_log = None
		relay = False

		for i in range(len(lst)):
			if prev_log == None:
				prev_log = lst[i][self.id]
			else:
				prev_log = lst[i-1][self.id]
			curr_log = lst[i][self.id]
			if curr_log != prev_log:
				dic[prev_log] = actions
				result.append(actions)
				actions = ''
				actions_sum += 'n '
			else:
				add_space = True
				prev_state = ''
				marker = lst[i][self.type]
				new_session = lst[i][self.session]
				if new_session:
					actions += 'ns '
					actions_sum += 'ns '
				if i > 0:
					prev_state = lst[i-1][self.type]
				if marker != 'p' and relay:
					relay = False
				if marker == 'p' and prev_state == 'p' and lst[i][self.cat] == lst[i-1][self.cat] and not relay:
					marker += 'c'
					relay = True
				if marker == 'p' and prev_state == 'p' and lst[i][self.cat] == lst[i-1][self.cat] and relay:
					marker = ''
				if marker == 'p' and prev_state == 'p' and lst[i][self.cat] != lst[i-1][self.cat]:
					marker = ''
				if marker != '':
					actions += marker
					actions_sum += marker
					actions += ' '
					actions_sum += ' '
		return result, dic, actions_sum


	def get_optimal_path(self, lst):
		dic = dict()
		result = []
		actions = ''
		actions_sum = ''

		prev_log = None
		relay = False

		for i in range(len(lst)):
			if prev_log == None:
				prev_log = lst[i][self.id]
			else:
				prev_log = lst[i-1][self.id]
			curr_log = lst[i][self.id]
			if curr_log != prev_log:
				dic[prev_log] = actions
				result.append(actions)
				actions = ''
				actions_sum += 'n '
			else:
				add_space = True
				prev_state = ''
				prev_page = '-1'
				curr_page = str(lst[i][self.cat])
				marker = lst[i][self.type]
				new_session = lst[i][self.session]
				if i > 0:
					prev_state = lst[i-1][self.type]
					prev_page = str(lst[i-1][self.cat])
				if new_session:
					actions += 'ns '
					actions_sum += 'ns '
					prev_state = 'ns'
				if marker != 'p' and relay:
					relay = False
				if marker == 'p' and prev_state == 'p' and curr_page == prev_page and not relay:
					marker += 'c'
					relay = True
				if marker == 'p' and prev_state == 'p' and curr_page == prev_page and relay:
					marker = ''
				if marker == 'p' and prev_state == 'p' and curr_page != prev_page:
					marker = ''
				if marker != '':
					actions += marker
					actions_sum += marker
					if curr_page == prev_page and marker != 'p' and marker != 'pc':
						actions += curr_page
						actions_sum += curr_page
					if curr_page != prev_page and marker != 'p' and marker != 'pc':
						actions += 'd'
						actions_sum += 'd'
					actions += ' '
					actions_sum += ' '
		return result, dic, actions_sum

	def get_full_path(self, lst):
		dic = dict()
		result = []
		actions = ''
		actions_sum = ''

		prev_log = None
		relay = False

		for i in range(len(lst)):
			if prev_log == None:
				prev_log = lst[i][self.id]
			else:
				prev_log = lst[i-1][self.id]
			curr_log = lst[i][self.id]
			if curr_log != prev_log:
				dic[prev_log] = actions
				result.append(actions)
				actions = ''
				actions_sum += 'n '
			else:
				add_space = True
				prev_state = ''
				marker = lst[i][self.type]
				new_session = lst[i][self.session]
				if new_session:
					actions += 'ns '
					actions_sum += 'ns '
				if i > 0:
					prev_state = lst[i-1][self.type]
				page = str(lst[i][self.cat])
				actions += marker + page + ' '
				actions_sum += marker + page +' '
		return result, dic, actions_sum

	def get_full_path_modified(self, lst):
		dic = dict()
		result = []
		actions = ''
		one_session_actions = ''
		actions_sum = ''

		prev_log = None
		prev_marker = None
		relay = False

		for i in range(len(lst)):
			if prev_log == None:
				prev_log = lst[i][self.id]
			else:
				prev_log = lst[i-1][self.id]
			curr_log = lst[i][self.id]
			if curr_log != prev_log:
				prev_marker = None
				if prev_log not in dic:
					dic[prev_log] = []
				dic[prev_log].append(one_session_actions)
				result.append(actions)
				actions = ''
				one_session_actions = ''
			else:
				add_space = True
				prev_state = ''
				marker = lst[i][self.type]
				new_session = lst[i][self.session]
				if new_session:
					prev_marker = None
					if curr_log not in dic:
						dic[curr_log] = []
					dic[curr_log].append(one_session_actions)
					one_session_actions = ''
				if i > 0:
					prev_state = lst[i-1][self.type]
				page = str(lst[i][self.cat])
				if marker + page + 'x' == prev_marker:
					continue

				if marker + page == prev_marker:
					page = page + 'x'

				actions += marker + page + ' '
				one_session_actions += marker + page + ' '
				actions_sum += marker + page +' '

				prev_marker = marker + page

		return result, dic, actions_sum

	def get_path(self, lst, mode = 'optimal'):
		if mode == 'optimal':
			return self.get_optimal_path(lst)
		elif mode == 'abbr':
			return self.get_abbr_path(lst)
		elif mode == 'full':
			return self.get_full_path(lst)
		elif mode == 'full_modified':
			return self.get_full_path_modified(lst)
		else:
			raise ValueError('mode not identified')

	def get_meanings(self):
		toPrint = []
		toPrint.append('n = new user')
		toPrint.append('ns = new session')
		toPrint.append('p = preview one/multiple category(s)')
		toPrint.append('pc = preview multiple products in the same category')
		toPrint.append('bd = buy an item in which the category from the previous action')
		toPrint.append('fd = add an item to favorites in which the category is different from the previous action')
		toPrint.append('cd = add an item to carts in which the category is different from the previous action')
		toPrint.append('b### = buy an item in which the category is the same with previous action, \
with following representing the category index')
		toPrint.append('f### = add an item to favorites in which the category is the same with previous action, \
with following representing the category index')
		toPrint.append('c### = add an item to carts in which the category is the same with previous action, with following representing the category index')
		toPrint.append('_space_ = distinguish different actions')
		toPrint.append('_elem = path of a single user (with the same user id)')
		for elem in toPrint:
			print(elem)

def partition_user_group(mappin_dic, category, user_dic):
	'''
	mapping_dic: key = numeric tag from KMeans, value = user_id
	category: key = numeric tag from KMeans, value = user group
	user_dic: key = user_id, value = user path
	'''
	result = []
	n_cat = max(category) + 1
	for i in range(n_cat): result.append(dict())
	for i in range(len(category)):
		user_id = mappin_dic[i]
		result[category[i]][user_id] = user_dic[user_id]
	return result

def build_ngram_from_string(sentence, n = 4, length = 50):
	tokenized = sentence.split()
	ngram_counts = Counter(ngrams(tokenized,n))
	freq = ngram_counts.most_common(length)
	return freq

def calculate_stat(ngram_freq, normalization = False, mode = 'page'):
	result = []
	for i in range(len(ngram_freq)):
		seq = ngram_freq[i][0]
		freq = ngram_freq[i][1]
		constructed = []
		for component in seq:
			if mode == 'page':
				page = component[1:]
				if page[-1] == 'x': page = page[:-1]
				constructed.append(page)
			if mode == 'behavior':
				behavior = component[0] + component[-1]
				if behavior[-1] != 'x': behavior = behavior[:-1]
				constructed.append(behavior)
		constructed = tuple(constructed)
		exist = False
		for j in range(len(result)):
			if result[j][0] == constructed:
				exist = True
				result[j][1] += freq
		if not exist:
			result.append([constructed, freq])
	result.sort(key = lambda x: x[1], reverse = True)
	if normalization:
		total = sum(freq for seq, freq in result)
		for i in range(len(result)):
			result[i][1] = result[i][1]/total
	return result

def generate_dictionary_from_stat(lst):
	result = dict()
	for i in range(len(lst)):
		seq, freq = lst[i][0], lst[i][1]
		result[seq] = freq
	return result

def generate_distribution_from_freq_dics(freq_dics):
	lst_result = []
	for i in range(len(freq_dics)):
		lst_result.append([])
	seq_key = []
	freq_dic = freq_dics[0]
	for seq in freq_dic:
		exist_all = True
		for freq_dic in freq_dics:
			if seq not in freq_dic:
				exist_all = False
		if exist_all:
			seq_key.append(seq)
			for i in range(len(freq_dics)):
				lst_result[i].append(freq_dics[i][seq])
	return lst_result, seq_key

def JS_divergence(p,q):
	M=(p+q)/2
	return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

def ngram_to_lst(freq):
	result = []
	used = []
	for i in range(len(freq)):
		temp = ''
		is_duplicate = False
		for k in range(len(used)):
			if set(used[k]) == set(freq[i][0]):
				is_duplicate = True
				break
		if is_duplicate:
			continue
		for j in range(len(freq[i][0])):
			used.append(freq[i][0])
			temp += freq[i][0][j]
			if j != len(freq[i][0]) - 1:
				temp += ' '
		result.append(temp)
	return result

def build_similarity(freq_str, dic):
	counter = 0
	result = {}
	for key in dic:
		counter += 1
		if counter == 500:
			print('|', end = '')
			counter = 0
		lst = []
		result[key] = lst
		match = dic[key]
		options = freq_str                
		for option in options:
			similarity, path = fuzz.ratio(match, option) / 100, option
			lst.append((similarity,path))            
	return result

def ld_to_ls(lst):
	'''
	Transform a list of dictionary to a list of string
	'''
	result = []
	for dic in lst:
		sentence = ''
		for key in dic:
			lst = dic[key]
			for elem in lst:
				sentence += elem
		result.append(sentence)
	return result

def get_sse_sil(x, kmax = 10):
	sil = []
	sse = []
	for k in range(2, kmax+1):
		print(k)
		kmeans = KMeans(n_clusters = k).fit(x)
		centroids = kmeans.cluster_centers_
		pred_clusters = kmeans.predict(x)
		curr_sse = 0
		for i in range(len(x)):
			curr_center = centroids[pred_clusters[i]]
			curr_sse += (x[i, 0] - curr_center[0]) ** 2 + (x[i, 1] - curr_center[1]) ** 2
		sse.append(curr_sse)     
		labels = kmeans.labels_
		sil.append(silhouette_score(model.docvecs.vectors_docs, labels, metric = 'euclidean'))
	return sse, sil

def plot_sse_sil(sil, sse):
	x = range(2, 2 + len(sil))
	y1 = sil
	y2 = sse
	plt.figure(figsize=(9, 3))
	plt.subplot(132)
	plt.plot(x, y1, 'bo')
	plt.title('Elbow')
	plt.subplot(133)
	plt.plot(x, y2, 'bo')
	plt.title('Silhouette')
	plt.show()

def get_sentences(dic):
	'''
	Input: user path dictionary
	Output: A list of lists where each inner list 
			represents a path (regardless of user id)
	'''
	result = []
	for key in dic:
		data = dic[key]
		for sentence in data:
			if sentence == '':
				continue
			else:
				tokenized = sentence.split()
				result.append(tokenized)
	return result

def get_sentences_doc(dic):
	'''
	Input: user path dictionary
	Output: a list of tuples and a mapping dictionary, 
		where each tuple is a numeric tag and concatenation of a user's path
		regardless of new sessions. The mapping dictionary maps tag to user_id
	'''
	result = []
	map_dic = {}
	i = 0
	for key in dic:
		words_bag = list()
		toAdd = (i, words_bag)
		map_dic[i] = key
		i += 1
		data = dic[key]
		for sentence in data:
			if sentence == '':
				continue
			else:
				tokenized = sentence.split()
				words_bag += tokenized
		result.append(toAdd)
	return result, map_dic

def get_documents(sentences_doc):
	'''
	Input: a sentence doc
	Output: a list of tagged documents for doc2vec.
	'''
	return [TaggedDocument(doc, [i]) for (i, doc) in sentences_doc]

def aggregate_group_path_data(dic, map_dic, l, group = 0):
	result = []
	for i in range(len(l)):
		cat = l[i]
		if cat != group and group != -1:
			continue
		else:
			user_id = map_dic[i]
			path = dic[user_id]
			for elem in path:
				if elem != '':
					result.append(elem[:-1])
	return result

def get_page_info(page_beh):
	if page_beh == '':
		return ''
	if page_beh[-1] == 'x':
		return page_beh[1:-1]
	else:
		return page_beh[1:]

def analyze_sequence_by_prior_sequence(path_data, mode = 'last', u_importance = 0.4, \
									   discard_same_page = False, discard_action = False, count_blank = True,
									  standardization = True):
	'''
	mode: first, last, constant, linear_decay, u
	'''
	if u_importance > 0.5:
		raise Exception('invalid u-shape importance')
	result = dict()
	for path_str in path_data:
		path = path_str.split()
		for i in range(len(path)):
			page_beh = path[i]
			# discard_action_modifier
			if discard_action:
				page_beh = get_page_info(page_beh)
			
			reason_elems = None
			reason_weights = None
			if page_beh not in result:
				result[page_beh] = list()
				result[page_beh].append([])
				result[page_beh].append([])
			if mode == 'last':
				reason_elems = [path[i-1]]
				reason_weights = [1]
			if mode == 'first':
				reason_elems = [path[0]]
				reason_weights = [1]
			if mode == 'constant':
				reason_elems = path[0:i]
				reason_weights = [1/len(reason_elems) for i in range(len(reason_elems))]
			if mode == 'linear_decay':
				reason_elems = path[0:i]
				reason_weights = [(i+1)/(sum(range(len(reason_elems)))+len(reason_elems)) for i in range(len(reason_elems))]
			if mode == 'u':
				reason_elems = path[0:i]
				if len(reason_elems) == 1:
					reason_weights = [1]
				elif len(reason_elems) == 2:
					reason_weights = [1/len(reason_elems) for i in range(len(reason_elems))]
				else:
					other_weight_sum = 1 - 2*u_importance
					reason_weights = [u_importance] + \
					[other_weight_sum/(len(reason_elems)-2) for i in range(len(reason_elems)-2)] + \
					[u_importance]                            
			if i == 0:
				# count_blank_modifier
				if not count_blank:
					continue
				
				reason_elems = ['']
				reason_weights = [1]
			for j in range(len(reason_elems)):
				reason_elem = reason_elems[j]
				reason_weight = reason_weights[j]
				# discard_same_page_modifier
				if discard_same_page and get_page_info(page_beh) == get_page_info(reason_elem):
					continue
				
				if reason_elem in result[page_beh][0]:
					index = result[page_beh][0].index(reason_elem)
					result[page_beh][1][index] = result[page_beh][1][index] + reason_weight
				else:
					result[page_beh][0].append(reason_elem)
					result[page_beh][1].append(reason_weight)
	for page_beh in result:
		result[page_beh][0] = [x for _,x in sorted(zip(result[page_beh][1],result[page_beh][0]), reverse = True)]
		weight_sum = sum(result[page_beh][1])
		if standardization:
			result[page_beh][1] = [weight/weight_sum for weight in result[page_beh][1]]
			result[page_beh][1] = sorted(result[page_beh][1], reverse = True)
		else:  
			result[page_beh][1] = sorted(result[page_beh][1], reverse = True)
	return result

def print_analysis(target, keys, length = 10, discard_same_page = True, discard_action = False, count_blank = False,
				  standardization = True):
	print('target: {}'.format(target))
	print()
	for key in keys:
		temp = analyze_sequence_by_prior_sequence(path_data, mode = key, \
						discard_same_page = discard_same_page, discard_action = discard_action, count_blank = count_blank, \
												 standardization = standardization)
		print('mode: {}'.format(key))
		print(temp[target][0][:length])
		print(temp[target][1][:length])
		print()

def generate_user_info_distribution(df, dftemp, inv_map, l, K = None):
	if K == None:
		print("Please pass the number of clusters as K")
		return None
	age_groups = []
	gender_groups = []
	age_lst = df['age'].unique().tolist()
	gender_lst = df['gender'].unique().tolist()
	user_id_lst = df['user_id'].unique().tolist()
	mapping = dict(zip(dftemp['user_id'], list(zip(dftemp['age'], dftemp['gender']))))
	for i in range(K):
		age_dic = dict()
		gender_dic = dict()
		for elem in age_lst:
			age_dic[elem] = 0
		for elem in gender_lst:
			gender_dic[elem] = 0
		age_groups.append(age_dic)
		gender_groups.append(gender_dic)
	for user_id in user_id_lst:
		cat_index = l[inv_map[user_id]]
		age = mapping[user_id][0]
		gender = mapping[user_id][1]
		age_groups[cat_index][age] = age_groups[cat_index][age] + 1
		gender_groups[cat_index][gender] = gender_groups[cat_index][gender] + 1
	return age_groups, gender_groups

def validate_reason_last(target, path_data, reason_dic, amount = 10):
	validation, total = 0, 0
	reasons = set(reason_dic[target][0][:amount])
	for single_session in path_data:
		sg_session = single_session.split()
		for i in range(len(sg_session) - 1):
			if sg_session[i] in reasons:
				total += 1
				if sg_session[i + 1] == target:
					validation += 1
		if (len(sg_session)) == 1 and '' in reasons:
			total, validation = total + 1, validation + 1
	if total != 0:
		print(target, end = '    Reasoning Coefficient: {}'.format(validation / total))
		print()
	if total == 0: total = -1
	return validation / total

def validate_reason_last_all(path_data, reason_dic):
	validate_dic = dict()
	for key in reason_dic: 
		validate_dic[key] = dict()
	validate_dic[''] = dict()
	for single_session in path_data:
		sg_session = single_session.split()
		if sg_session[0] not in validate_dic['']:
			validate_dic[''][sg_session[0]] = 0
		validate_dic[''][sg_session[0]] += 1
		for i in range(len(sg_session) - 1):
			if sg_session[i+1] not in validate_dic[sg_session[i]]:
				validate_dic[sg_session[i]][sg_session[i+1]] = 0
			validate_dic[sg_session[i]][sg_session[i+1]] += 1
	macro = []
	weights = []
	for key in validate_dic:
		counts = list(validate_dic[key].values())
		if (len(counts) == 0): continue
		total = sum(counts)
		temp = []
		for count in counts:
			temp.append(count/total)
		macro.append(np.average(temp, weights=counts))
		weights.append(total)
	return np.average(macro), np.average(macro, weights=weights)

def group_age(x):
	if x > 17 and x <= 23:
		return 0
	elif x > 23 and x <= 27:
		return 1
	elif x > 27 and x <= 31:
		return 2
	elif x > 31 and x <= 38:
		return 3
	else:
		return 4

def validation_report(path_data_sublst, reason_dic_sublst, title = None, K = None, globe = True):
	if K == None:
		print("Please pass the number of clusters as K")
		return None
	print("---------------------------------------------------------------------")
	if title != None: print(title)
	start = -1 * int(globe)
	mac_globe = 0
	wei_globe = 0
	mac_lst = []
	wei_lst = []
	for i in range(start, K):
		path_data = path_data_sublst[i + 1]
		reason_dic = reason_dic_sublst[i + 1]
		mac, wei = validate_reason_last_all(path_data, reason_dic)
		if i == -1:
			mac_globe = mac
			wei_globe = wei
		if i >= 0:
			mac_lst.append(mac)
			wei_lst.append(wei)
		print("Group {} | Macro Accuracy: {} | Weighted Accuracy: {}".format(i, mac, wei))
	print()
	if globe:
		print("Without Clustering:")
		print("Macro Accuracy: {} | Weighted Accuracy: {}".format(mac_globe, wei_globe))
	print()
	print("With Clstering:")
	print("Macro Accuracy: {} | Weighted Accuracy: {}".format(np.average(mac_lst), np.average(wei_lst)))
	print()
	if globe:
		return mac_globe, wei_globe, np.average(mac_lst), np.average(wei_lst)
	else:
		return np.average(mac_lst), np.average(wei_lst)

def generate_embed(l, inv_map, df):
	lst = df['user_id'].tolist()
	result = []
	for elem in lst:
		result.append(l[inv_map[elem]])
	return np.array(result)

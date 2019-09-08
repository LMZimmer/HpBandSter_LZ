import logging
from copy import deepcopy
import traceback
import math


import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import statsmodels.api as sm

from hpbandster.core.base_config_generator import base_config_generator


class BOHB_Multi_KDE(base_config_generator):
	def __init__(self, configspace, min_points_in_model = None,
				top_n_percent=15, num_samples = 64, random_fraction=1/3,
				bandwidth_factor=3, min_bandwidth=1e-3, budgets=None, eta=3,
				n_kdes=1, **kwargs):
		"""
			Fits for each given budget a kernel density estimator on the best N percent of the
			evaluated configurations on this budget.


			Parameters:
			-----------
			configspace: ConfigSpace
				Configuration space object
			top_n_percent: int
				Determines the percentile of configurations that will be used as training data
				for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
				for training.
			min_points_in_model: int
				minimum number of datapoints needed to fit a model
			num_samples: int
				number of samples drawn to optimize EI via sampling
			random_fraction: float
				fraction of random configurations returned
			bandwidth_factor: float
				widens the bandwidth for contiuous parameters for proposed points to optimize EI
			min_bandwidth: float
				to keep diversity, even when all (good) samples have the same value for one of the parameters,
				a minimum bandwidth (Default: 1e-3) is used instead of zero. 

		"""
		super().__init__(**kwargs)
		self.budgets = budgets
		self.top_n_percent=top_n_percent
		self.configspace = configspace
		self.bw_factor = bandwidth_factor
		self.min_bandwidth = min_bandwidth
		self.max_kdes = n_kdes

		self.min_points_in_model = min_points_in_model
		if min_points_in_model is None:
			self.min_points_in_model = len(self.configspace.get_hyperparameters())+1
		
		if self.min_points_in_model < len(self.configspace.get_hyperparameters())+1:
			self.logger.warning('Invalid min_points_in_model value. Setting it to %i'%(len(self.configspace.get_hyperparameters())+1))
			self.min_points_in_model =len(self.configspace.get_hyperparameters())+1
		
		self.num_samples = num_samples
		self.random_fraction = random_fraction

		hps = self.configspace.get_hyperparameters()

		self.kde_vartypes = ""
		self.vartypes = []


		for h in hps:
			if hasattr(h, 'sequence'):
				raise RuntimeError('This version on BOHB does not support ordinal hyperparameters. Please encode %s as an integer parameter!'%(h.name))
			
			if hasattr(h, 'choices'):
				self.kde_vartypes += 'u'
				self.vartypes +=[ len(h.choices)]
			else:
				self.kde_vartypes += 'c'
				self.vartypes +=[0]
		
		self.vartypes = np.array(self.vartypes, dtype=int)

		# store precomputed probs for the categorical parameters
		self.cat_probs = []
		

                # initialize as many KDEs as there are successive halfing iterations
		print("CG: INITIALIZING BOHB_MULTI_KDE. CREATING", n_kdes, "KDE MODELS...")					# HERE
		self.configs = np.array([dict() for i in range(self.max_kdes)])
		self.losses = np.array([dict() for i in range(self.max_kdes)])
		self.good_config_rankings = np.array([dict() for i in range(self.max_kdes)])
		self.kde_models = np.array([dict() for i in range(self.max_kdes)])
		self.last_permutation = [idx for idx in range(len(self.kde_models))]
		print("CG: INITIALIZED CONFIGS", self.configs)
		print("CG: INITIALIZED LOSSES", self.losses)
		print("CG: INITIALIZED KDE MODELS", self.kde_models)

	def permute_kdes(self, permutation):
		print("CG: KDE Models before permutation:", self.kde_models)
		self.invert_permutations()
		self.configs = self.configs[permutation]
		self.losses = self.losses[permutation]
		self.good_config_rankings = self.good_config_rankings[permutation]
		self.kde_models = self.kde_models[permutation]
		print("CG: KDE Models after permutation:", self.kde_models)

		self.last_permutation = permutation

	def invert_permutations(self):	# Need to revert it first such that it is the same as APT ...
		perm_inv = self.get_inverse_permutation(self.last_permutation)
		self.configs = self.configs[perm_inv]
		self.losses = self.losses[perm_inv]
		self.good_config_rankings = self.good_config_rankings[perm_inv]
		self.kde_models = self.kde_models[perm_inv]

	def get_inverse_permutation(self, perm):
		inverse = [0] * len(perm)
		for i,p in enumerate(perm):
			inverse[p] = i
		return inverse

	def get_config(self, budget):
		"""
			Function to sample a new configuration

			This function is called inside Hyperband to query a new configuration


			Parameters:
			-----------
			budget: float
				the budget for which this configuration is scheduled

			returns: config
				should return a valid configuration

		"""
		
		self.logger.debug('start sampling a new configuration.')

		n_kdes = self.max_kdes

		# UNCOMMENT THIS TO ONLY SAMPLE FROM KDES FOR ACTIVE DATASETS
		# Infer number of KDEs to use at current budget (mimic MultiAutoPyTorch implementation)
		#max_datasets = self.max_kdes
		#max_steps = len(self.budgets)
		#current_step = max_steps - int((math.log(max(self.budgets)) - math.log(budget)) / math.log(3)) if budget > 1e-10 else 0
		#n_kdes = math.floor(math.pow(max_datasets, current_step/max(1, max_steps)) + 1e-10)

		print("CG: SAMPLING CONFIG FOR BUDGET", budget, "... USING", n_kdes, "KDE MODELS")										# HERE

		sample = None
		info_dict = {}
		
		# If no model is available, sample from prior
		# also mix in a fraction of random configs
		no_kde_models = True
		for ind in range(n_kdes):					# check if there are kde models
			if len(self.kde_models[ind].keys()) != 0:
				no_kde_models = False
				break
		if no_kde_models or np.random.rand() < self.random_fraction:
			print("CG: No KDE models found and/or sampling randomly")												# HERE
			sample =  self.configspace.sample_configuration()
			info_dict['model_based_pick'] = False

		best = np.inf
		best_vector = None

		if sample is None:
			try:
				# Create objective function (g/l), accumulate seen configs/datums
				l = []
				g = []
				kde_good = []
				kde_bad = []
				for ind in range(n_kdes):
					if len(self.kde_models[ind].keys()) > 0:									# Add if dataset has a kde model
						print("CG: APPENDING KDE MODEl", ind)									# HERE
						budget = max(self.kde_models[ind].keys())								# Get max budget model
						l.append(self.kde_models[ind][budget]['good'].pdf)
						g.append(self.kde_models[ind][budget]['bad' ].pdf)
						kde_good.append(self.kde_models[ind][budget]['good'])
						kde_bad.append(self.kde_models[ind][budget]['bad'])

				mean_g = lambda x: np.mean(np.array([func.__call__(x) for func in g]))
				mean_l = lambda x: np.mean(np.array([func.__call__(x) for func in l]))
				minimize_me = lambda x: max(1e-32,mean_g(x))/max(mean_l(x),1e-32)							# g/l becomes mean over all g/ mean over all l

				# CHANGE KDE_GOOD TO KDES_GOOOD ABOVE (*2)
				# Accumulate data, bws
				datums = []
				bws = []
				for kde in kde_good:
					print("CG: KDE DATA", kde.data)											# HERE
					print("CG: KDE BW", kde.bw)											# HERE
					for datum in kde.data:
						if datum not in datums:			# append data and bw without duplicates
							datums.append(datum)
							bws.append(kde.bw)
				idx = np.random.randint(0, len(datums))
				datums = datums[idx]					# shuffle, as array?


				# Sample num_samples
				for i in range(self.num_samples):
					idx = np.random.randint(0, len(kde_good[0].data))								# delete
					datum = kde_good[0].data[idx]											# delete
					if i==0:
						print("CG: DATUM:", datum, ", BW:", kde_good[0].bw)							# HERE
					vector = []
					
					for m,bw,t in zip(datum, kde_good[0].bw, self.vartypes):							# datum -> datums, bw -> bws?
						
						bw = max(bw, self.min_bandwidth)
						if t == 0:
							bw = self.bw_factor*bw
							try:
								vector.append(sps.truncnorm.rvs(-m/bw,(1-m)/bw, loc=m, scale=bw))
							except:
								self.logger.warning("Truncated Normal failed for:\ndatum=%s\nbandwidth=%s\nfor entry with value %s"%(datum, kde_good[0].bw, m))
								self.logger.warning("data in the KDE:\n%s"%kde_good[0].data)
						else:
							
							if np.random.rand() < (1-bw):
								vector.append(int(m))
							else:
								vector.append(np.random.randint(t))
					val = minimize_me(vector)

					if not np.isfinite(val):
						self.logger.warning('sampled vector: %s has EI value %s'%(vector, val))
						self.logger.warning("data in the KDEs:\n%s\n%s"%(kde_good[0].data, kde_bad[0].data))
						self.logger.warning("bandwidth of the KDEs:\n%s\n%s"%(kde_good[0].bw, kde_bad[0].bw))
						self.logger.warning("l(x) = %s"%(l(vector)))
						self.logger.warning("g(x) = %s"%(g(vector)))

						# right now, this happens because a KDE does not contain all values for a categorical parameter
						# this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this one
						# if the good_kde has a finite value, i.e. there is no config with that value in the bad kde, so it shouldn't be terrible.
						if np.isfinite(l(vector)):
							best_vector = vector
							break

					if val < best:
						best = val
						best_vector = vector

				if best_vector is None:
					self.logger.debug("Sampling based optimization with %i samples failed -> using random configuration"%self.num_samples)
					sample = self.configspace.sample_configuration().get_dictionary()
					info_dict['model_based_pick']  = False
				else:
					self.logger.debug('best_vector: {}, {}, {}, {}'.format(best_vector, best, l(best_vector), g(best_vector)))
					for i, hp_value in enumerate(best_vector):
						if isinstance(
							self.configspace.get_hyperparameter(
								self.configspace.get_hyperparameter_by_idx(i)
							),
							ConfigSpace.hyperparameters.CategoricalHyperparameter
						):
							best_vector[i] = int(np.rint(best_vector[i]))
					sample = ConfigSpace.Configuration(self.configspace, vector=best_vector).get_dictionary()
					
					try:
						sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
									configuration_space=self.configspace,
									configuration=sample
									)
						info_dict['model_based_pick'] = True

					except Exception as e:
						self.logger.warning(("="*50 + "\n")*3 +\
								"Error converting configuration:\n%s"%sample+\
								"\n here is a traceback:" +\
								traceback.format_exc())
						raise(e)

			except:
				self.logger.warning("Sampling based optimization with %i samples failed\n %s \nUsing random configuration"%(self.num_samples, traceback.format_exc()))
				sample = self.configspace.sample_configuration()
				info_dict['model_based_pick']  = False


		try:
			sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
				configuration_space=self.configspace,
				configuration=sample.get_dictionary()
			).get_dictionary()
		except Exception as e:
			self.logger.warning("Error (%s) converting configuration: %s -> "
								"using random configuration!",
								e,
								sample)
			sample = self.configspace.sample_configuration().get_dictionary()
		self.logger.debug('done sampling a new configuration.')
		return sample, info_dict


	def impute_conditional_data(self, array):

		return_array = np.empty_like(array)

		for i in range(array.shape[0]):
			datum = np.copy(array[i])
			nan_indices = np.argwhere(np.isnan(datum)).flatten()

			while (np.any(nan_indices)):
				nan_idx = nan_indices[0]
				valid_indices = np.argwhere(np.isfinite(array[:,nan_idx])).flatten()

				if len(valid_indices) > 0:
					# pick one of them at random and overwrite all NaN values
					row_idx = np.random.choice(valid_indices)
					datum[nan_indices] = array[row_idx, nan_indices]

				else:
					# no good point in the data has this value activated, so fill it with a valid but random value
					t = self.vartypes[nan_idx]
					if t == 0:
						datum[nan_idx] = np.random.rand()
					else:
						datum[nan_idx] = np.random.randint(t)

				nan_indices = np.argwhere(np.isnan(datum)).flatten()
			return_array[i,:] = datum
		return(return_array)

	def new_result(self, job, update_model=True):
		"""
			function to register finished runs

			Every time a run has finished, this function should be called
			to register it with the result logger. If overwritten, make
			sure to call this method from the base class to ensure proper
			logging.


			Parameters:
			-----------
			job: hpbandster.distributed.dispatcher.Job object
				contains all the info about the run
		"""

		super().new_result(job)

		budget = job.kwargs["budget"]

		if job.result is None:
			# One could skip crashed results, but we decided to
			# assign a +inf loss and count them as bad configurations
			losses = [np.inf]*self.max_kdes
		else:
			# same for non numeric losses.
			# Note that this means losses of minus infinity will count as bad!
			losses = job.result["losses"] if np.isfinite(job.result["loss"]) else [np.inf]*self.max_kdes
			print("CG: REGISTER NEW RESULT FOR BUDGET", budget, "... FOUND LOSSES", losses)								# HERE

		# update as many kdes as there are losses
		n_kdes = len(losses)

		# if the budget has not been seen by a new kde, initialize its configs dict with an empty list
		for ind in range(n_kdes):
			if budget not in self.configs[ind].keys():
				self.configs[ind][budget] = []
				self.losses[ind][budget] = []													# careful losses is not self.losses
				print("CG: BUDGET", budget, "NOT FOUND IN KDE", ind, ", INITIALIZING TO", self.losses[ind])					# HERE


		# We want to get a numerical representation of the configuration in the original space
		conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])

		for ind in range(n_kdes):

			# skip model building if we already have a bigger model
			if max(list(self.kde_models[ind].keys()) + [-np.inf]) > budget:
				print("CG: SKIPPING KDE MODEL BUILDING FOR MODEL", ind)										# HERE
				if ind == n_kdes-1:
					return
				else:
					continue

			self.configs[ind][budget].append(conf.get_array())
			self.losses[ind][budget].append(losses[ind])
			print("CG: APPENDING LOSSES TO COLLECTION:", self.losses)										# HERE

			# skip model building:
			#		a) if not enough points are available
			if len(self.configs[ind][budget]) <= self.min_points_in_model-1:
				self.logger.debug("Only %i run(s) for budget %f available, need more than %s -> can't build model!"%(len(self.configs[ind][budget]), budget, self.min_points_in_model+1))
				if ind==n_kdes-1:
					return
				else:
					continue

			#		b) during warm starting when we feed previous results in and only update once
			if not update_model:
				if ind==n_kdes-1:
					return
				else:
					continue

			train_configs = np.array(self.configs[ind][budget])
			train_losses =  np.array(self.losses[ind][budget])
			print("CG: UPDATING MODEL", ind, "WITH LOSS", train_losses)										# HERE

			n_good= max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0])//100 )
			#n_bad = min(max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100), 10)
			n_bad = max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100)

			# Refit KDEs for the current budget
			idx = np.argsort(train_losses)

			train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
			train_data_bad  = self.impute_conditional_data(train_configs[idx[n_good:n_good+n_bad]])

			if train_data_good.shape[0] <= train_data_good.shape[1]:
				return
			if train_data_bad.shape[0] <= train_data_bad.shape[1]:
				return
		
			#more expensive crossvalidation method
			#bw_estimation = 'cv_ls'

			# quick rule of thumb
			bw_estimation = 'normal_reference'

			bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad,  var_type=self.kde_vartypes, bw=bw_estimation)
			good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde_vartypes, bw=bw_estimation)

			bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth,None)
			good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth,None)

			self.kde_models[ind][budget] = {
							'good': good_kde,
							'bad' : bad_kde
							}

			# update probs for the categorical parameters for later sampling
			self.logger.debug('done building a new model for budget %f based on %i/%i split\nBest loss for this budget:%f\n\n\n\n\n'%(budget, n_good, n_bad, np.min(train_losses)))


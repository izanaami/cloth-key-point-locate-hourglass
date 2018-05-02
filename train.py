import configparser
from hourglass_tiny import HourglassModel
from datagen import DataGenerator


def process_config(conf_file):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


g_train_data_file = '/home/jxy/Documents/服饰关键点定位/data/train/Annotations/train.csv'
g_warmup_data_file = '/home/jxy/Documents/服饰关键点定位/data/train_warmup/Annotations/annotations.csv'

img_dir = '/home/jxy/Documents/服饰关键点定位/data/train/'
warmup_dir = '/home/jxy/Documents/服饰关键点定位/data/train_warmup/'
batch_size = 8
nEpochs = 50
epoch_size = 1000
learning_rate = 0.0005
learning_rate_decay = 0.96
decay_step = 2000
nFeats = 256
nStacks = 4
nModules = 1
nLow = 4
dropout_rate = 0.2
saver_step = 500
weight_loss = True
validation_rate = 0.01

def training_all(model_name=None):
	training_blouse(model_name=model_name)
	training_dress(model_name=model_name)
	training_outwear(model_name=model_name)
	training_skirt(model_name=model_name)
	training_trousers(model_name=model_name)

	print('all done.')


def training_blouse(model_name=None):
	dataset_blouse = DataGenerator(train_data_file=g_train_data_file, warmup_data_file=g_warmup_data_file)

	print('--Parsing Config File')
	params = process_config('config_blouse.cfg')

	print('--' + params['category'] + 'training begin:')
	dataset_blouse._get_img_dir(img_dir=img_dir, warmup_dir=warmup_dir)
	dataset_blouse._get_joints_name(joints_name=params['joint_list'],
									joints_num=params['num_joints'])
	dataset_blouse._read_img_dir_from_csv()
	dataset_blouse._create_train_table(category=params['category'])
	dataset_blouse._randomize()
	dataset_blouse._create_sets(validation_rate=validation_rate)

	model_blouse = HourglassModel(nFeat=nFeats,
								  nStack=nStacks,
								  nModules=nModules,
								  nLow=nLow,
								  outputDim=params['num_joints'],
								  batch_size=batch_size,
								  attention=False,
								  training=True,
								  drop_rate=dropout_rate,
								  lear_rate=learning_rate,
								  decay=learning_rate_decay,
								  decay_step=decay_step,
								  dataset=dataset_blouse,
								  name=params['name'] + model_name,
								  logdir_train=params['log_dir_train'],
								  logdir_test=params['log_dir_test'],
								  tiny=False,
								  w_loss=weight_loss,
								  joints=params['joint_list'],
								  modif=False)
	model_blouse.generate_model()
	model_blouse.training_init(nEpochs=nEpochs,
							   epochSize=epoch_size,
							   saveStep=saver_step,
							   dataset=None,
							   category=params['category'])

	del dataset_blouse
	del model_blouse


def training_dress(model_name=None):
	dataset_dress = DataGenerator(train_data_file=g_train_data_file, warmup_data_file=g_warmup_data_file)

	print('--Parsing Config File')
	params = process_config('config_dress.cfg')

	print('--' + params['category'] + 'training begin:')
	dataset_dress._get_img_dir(img_dir=img_dir, warmup_dir=warmup_dir)
	dataset_dress._get_joints_name(joints_name=params['joint_list'],
									joints_num=params['num_joints'])
	dataset_dress._read_img_dir_from_csv()
	dataset_dress._create_train_table(category=params['category'])
	dataset_dress._randomize()
	dataset_dress._create_sets(validation_rate=validation_rate)

	model_dress = HourglassModel(nFeat=nFeats,
								  nStack=nStacks,
								  nModules=nModules,
								  nLow=nLow,
								  outputDim=params['num_joints'],
								  batch_size=batch_size,
								  attention=False,
								  training=True,
								  drop_rate=dropout_rate,
								  lear_rate=learning_rate,
								  decay=learning_rate_decay,
								  decay_step=decay_step,
								  dataset=dataset_dress,
								  name=params['name'] + model_name,
								  logdir_train=params['log_dir_train'],
								  logdir_test=params['log_dir_test'],
								  tiny=False,
								  w_loss=weight_loss,
								  joints=params['joint_list'],
								  modif=False)
	model_dress.generate_model()
	model_dress.training_init(nEpochs=nEpochs,
							   epochSize=epoch_size,
							   saveStep=saver_step,
							   dataset=None,
							   category=params['category'])

	del dataset_dress
	del model_dress


def training_outwear(model_name=None):
	dataset_outwear = DataGenerator(train_data_file=g_train_data_file, warmup_data_file=g_warmup_data_file)

	print('--Parsing Config File')
	params = process_config('config_outwear.cfg')

	print('--' + params['category'] + 'training begin:')
	dataset_outwear._get_img_dir(img_dir=img_dir, warmup_dir=warmup_dir)
	dataset_outwear._get_joints_name(joints_name=params['joint_list'],
									joints_num=params['num_joints'])
	dataset_outwear._read_img_dir_from_csv()
	dataset_outwear._create_train_table(category=params['category'])
	dataset_outwear._randomize()
	dataset_outwear._create_sets(validation_rate=validation_rate)

	model_outwear = HourglassModel(nFeat=nFeats,
								  nStack=nStacks,
								  nModules=nModules,
								  nLow=nLow,
								  outputDim=params['num_joints'],
								  batch_size=batch_size,
								  attention=False,
								  training=True,
								  drop_rate=dropout_rate,
								  lear_rate=learning_rate,
								  decay=learning_rate_decay,
								  decay_step=decay_step,
								  dataset=dataset_outwear,
								  name=params['name'] + model_name,
								  logdir_train=params['log_dir_train'],
								  logdir_test=params['log_dir_test'],
								  tiny=False,
								  w_loss=weight_loss,
								  joints=params['joint_list'],
								  modif=False)
	model_outwear.generate_model()
	model_outwear.training_init(nEpochs=nEpochs,
							   epochSize=epoch_size,
							   saveStep=saver_step,
							   dataset=None,
							   category=params['category'])

	del dataset_outwear
	del model_outwear


def training_skirt(model_name=None):
	dataset_skirt = DataGenerator(train_data_file=g_train_data_file, warmup_data_file=g_warmup_data_file)

	print('--Parsing Config File')
	params = process_config('config_skirt.cfg')

	print('--' + params['category'] + 'training begin:')
	dataset_skirt._get_img_dir(img_dir=img_dir, warmup_dir=warmup_dir)
	dataset_skirt._get_joints_name(joints_name=params['joint_list'],
									joints_num=params['num_joints'])
	dataset_skirt._read_img_dir_from_csv()
	dataset_skirt._create_train_table(category=params['category'])
	dataset_skirt._randomize()
	dataset_skirt._create_sets(validation_rate=validation_rate)

	model_skirt = HourglassModel(nFeat=nFeats,
								  nStack=nStacks,
								  nModules=nModules,
								  nLow=nLow,
								  outputDim=params['num_joints'],
								  batch_size=batch_size,
								  attention=False,
								  training=True,
								  drop_rate=dropout_rate,
								  lear_rate=learning_rate,
								  decay=learning_rate_decay,
								  decay_step=decay_step,
								  dataset=dataset_skirt,
								  name=params['name'] + model_name,
								  logdir_train=params['log_dir_train'],
								  logdir_test=params['log_dir_test'],
								  tiny=False,
								  w_loss=weight_loss,
								  joints=params['joint_list'],
								  modif=False)
	model_skirt.generate_model()
	model_skirt.training_init(nEpochs=nEpochs,
							   epochSize=epoch_size,
							   saveStep=saver_step,
							   dataset=None,
							   category=params['category'])

	del dataset_skirt
	del model_skirt


def training_trousers(model_name=None):
	dataset_trousers = DataGenerator(train_data_file=g_train_data_file, warmup_data_file=g_warmup_data_file)

	print('--Parsing Config File')
	params = process_config('config_trousers.cfg')

	print('--' + params['category'] + 'training begin:')
	dataset_trousers._get_img_dir(img_dir=img_dir, warmup_dir=warmup_dir)
	dataset_trousers._get_joints_name(joints_name=params['joint_list'],
									joints_num=params['num_joints'])
	dataset_trousers._read_img_dir_from_csv()
	dataset_trousers._create_train_table(category=params['category'])
	dataset_trousers._randomize()
	dataset_trousers._create_sets(validation_rate=validation_rate)

	model_trousers = HourglassModel(nFeat=nFeats,
								  nStack=nStacks,
								  nModules=nModules,
								  nLow=nLow,
								  outputDim=params['num_joints'],
								  batch_size=batch_size,
								  attention=False,
								  training=True,
								  drop_rate=dropout_rate,
								  lear_rate=learning_rate,
								  decay=learning_rate_decay,
								  decay_step=decay_step,
								  dataset=dataset_trousers,
								  name=params['name'] + model_name,
								  logdir_train=params['log_dir_train'],
								  logdir_test=params['log_dir_test'],
								  tiny=False,
								  w_loss=weight_loss,
								  joints=params['joint_list'],
								  modif=False)
	model_trousers.generate_model()
	model_trousers.training_init(nEpochs=nEpochs,
							   epochSize=epoch_size,
							   saveStep=saver_step,
							   dataset=None,
							   category=params['category'])

	del dataset_trousers
	del model_trousers


if __name__ == '__main__':
	modelname = '_mse'

	# training_all(model_name=modelname)

	# training_blouse(model_name=modelname)
	# training_dress(model_name=modelname)
	# training_outwear(model_name=modelname)
	# training_skirt(model_name=modelname)
	training_trousers(model_name=modelname)

	print('all done.')
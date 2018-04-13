import configparser
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
import csv
from inference import Inference
import cv2

g_train_data_file = '/home/jxy/Desktop/服饰关键点定位/data/train/Annotations/train.csv'
g_test_data_file = '/home/jxy/Desktop/服饰关键点定位/data/test/test.csv'

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
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


def training():
	print('--Creating Dataset')
	# dataset1 = DataGenerator(train_data_file=g_train_data_file)
	# dataset2 = DataGenerator(train_data_file=g_train_data_file)
	# dataset3 = DataGenerator(train_data_file=g_train_data_file)
	# dataset4 = DataGenerator(train_data_file=g_train_data_file)
	dataset5 = DataGenerator(train_data_file=g_train_data_file)

	# training_separately(dataset1, 'config_blouse.cfg')
	# training_separately(dataset2, 'config_dress.cfg')
	# training_separately(dataset3, 'config_outwear.cfg')
	# training_separately(dataset4, 'config_skirt.cfg')
	training_separately(dataset5, 'config_trousers.cfg')

	print('all done.')


def training_separately(dataset=None, config_file=None):
	print('--Parsing Config File')
	params = process_config(config_file)

	print('--' + params['category'] + 'training begin:')

	dataset._get_img_dir(img_dir=params['img_dir'])
	dataset._get_joints_name(joints_name=params['joint_list'],
							 joints_num=params['num_joints'])
	dataset._create_train_table(category=params['category'])
	dataset._randomize()
	dataset._create_sets(validation_rate=0.1)

	model = HourglassModel(nFeat=params['nfeats'],
								  nStack=params['nstacks'],
								  nModules=params['nmodules'],
								  nLow=params['nlow'],
								  outputDim=params['num_joints'],
								  batch_size=params['batch_size'],
								  attention=params['mcam'],
								  training=True,
								  drop_rate=params['dropout_rate'],
								  lear_rate=params['learning_rate'],
								  decay=params['learning_rate_decay'],
								  decay_step=params['decay_step'],
								  dataset=dataset,
								  name=params['name'],
								  logdir_train=params['log_dir_train'],
								  logdir_test=params['log_dir_test'],
								  tiny=params['tiny'],
								  w_loss=params['weighted_loss'],
								  joints=params['joint_list'],
								  modif=False)
	model.generate_model()
	model.training_init(nEpochs=params['nepochs'],
							   epochSize=params['epoch_size'],
							   saveStep=params['saver_step'],
							   dataset=None,
							   category=params['category'])


def testing():
	# 将要写入csv的内容
	write_datas = [['image_id',
					'image_category',
					'neckline_left',
					'neckline_right',
					'center_front',
					'shoulder_left',
					'shoulder_right',
					'armpit_left',
					'armpit_right',
					'waistline_left',
					'waistline_right',
					'cuff_left_in',
					'cuff_left_out',
					'cuff_right_in',
					'cuff_right_out',
					'top_hem_left',
					'top_hem_right',
					'waistband_left',
					'waistband_right',
					'hemline_left',
					'hemline_right',
					'crotch',
					'bottom_left_in',
					'bottom_left_out',
					'bottom_right_in',
					'bottom_right_out']]

	# 读取模型
	inf_blouse = Inference(config_file='config_blouse.cfg', model='model_blouse_80', yoloModel=None)
	inf_dress = Inference(config_file='config_dress.cfg', model='model_dress_80', yoloModel=None)
	inf_outwear = Inference(config_file='config_outwear.cfg', model='model_outwear_80', yoloModel=None)
	inf_skirt = Inference(config_file='config_skirt.cfg', model='model_skirt_80', yoloModel=None)
	inf_trousers = Inference(config_file='config_trousers.cfg', model='model_trousers_80', yoloModel=None)

	# 读取csv得到图片名称和种类
	imgs = []
	categorys = []
	with open(g_test_data_file) as csv_file:
		reader = csv.reader(csv_file)  # 逐行读取
		for row in reader:
			imgs.append(row[0])
			categorys.append(row[1])
	del (imgs[0])  # 删除第一行
	del (categorys[0])

	# 图片完整路径
	img_path = '/home/jxy/Desktop/服饰关键点定位/data/test/'
	images = []
	for name in imgs:
		img_full_path = img_path + name
		images.append(img_full_path)

	# for i in range(2):  # 先用两张图片测试
	for i in range(len(images)):
		print(str(i) + '/' + str(len(images)))
		line = [imgs[i], categorys[i]]

		# 读入图片并resize
		img, h, w = open_img(images[i], color='RGB')
		img = cv2.resize(img, (256, 256))
		if categorys[i] == 'blouse':
			predict_joints = inf_blouse.predictJoints(img, mode='gpu', thresh=0.2)
		elif categorys[i] == 'dress':
			predict_joints = inf_dress.predictJoints(img, mode='gpu', thresh=0.2)
		elif categorys[i] == 'outwear':
			predict_joints = inf_outwear.predictJoints(img, mode='gpu', thresh=0.2)
		elif categorys[i] == 'skirt':
			predict_joints = inf_skirt.predictJoints(img, mode='gpu', thresh=0.2)
		elif categorys[i] == 'trousers':
			predict_joints = inf_trousers.predictJoints(img, mode='gpu', thresh=0.2)

		# 对预测得到的joints进行处理
		coordinates = process_joints(predict_joints, categorys[i], h, w)
		line = line + coordinates  # 名称+类别+坐标
		write_datas.append(line)

	# 写入csv
	write_path = '/home/jxy/Desktop/服饰关键点定位/data/test.csv'
	with open(write_path, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerows(write_datas)


def open_img(path, color='RGB'):
	""" Open an image
	Args:
		name	: Name of the sample    # Images/blouse/02b54c183d2dbd2c056db14303064886.jpg
		color	: Color Mode (RGB/BGR/GRAY)
	"""
	img = cv2.imread(path)
	if color == 'RGB':
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img, img.shape[0], img.shape[1]
	elif color == 'BGR':
		return img, img.shape[0], img.shape[1]
	elif color == 'GRAY':
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img, img.shape[0], img.shape[1]
	else:
		print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')


def process_joints(joints, category, h, w):
	blouse = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
	skirt = [15, 16, 17, 18]
	outwear = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
	dress = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18]
	trousers = [15, 16, 19, 20, 21, 22, 23]

	line = []
	if category == 'blouse':
		index = blouse
	elif category == 'skirt':
		index = skirt
	elif category == 'outwear':
		index = outwear
	elif category == 'dress':
		index = dress
	elif category == 'trousers':
		index = trousers
	else:
		print('category error')
		exit(-1)

	for i in range(24):
		flag = 0
		for j in range(len(index)):
			if i == index[j]:
				# 256*256大小下的坐标
				x = int(joints[j][1])
				y = int(joints[j][0])
				x = int(x * w / 256)
				y = int(y * h / 256)
				coordinate = str(x) + '_' + str(y) + '_1'
				line.append(coordinate)
				flag = 1
				break
		if flag == 0:
			line.append('-1_-1_-1')

	return line


def test_1():
	img_path = '/home/jxy/Desktop/服饰关键点定位/data/test/Images/blouse/'
	name = '0a02d1f1a99aede99cca75b45efc5910.jpg'
	name = img_path + name
	inf = Inference(config_file='config_blouse.cfg',
					model='model_blouse_80',
					yoloModel=None)

	img, h, w = open_img(name, color='RGB')
	img = cv2.resize(img, (256, 256))
	predict_joints = inf.predictJoints(img, mode='gpu', thresh=0.2)  # output type: np.array

	for i in range(len(predict_joints)):
		cv2.circle(img, (int(predict_joints[i][1]), int(predict_joints[i][0])), 1, (0, 0, 255), 2)

	#cv2.circle(img, (int(predict_joints[2][1]), int(predict_joints[2][0])), 1, (0, 0, 255), 2)
	cv2.imshow('x', img)
	cv2.waitKey()


if __name__ == '__main__':
	testing()
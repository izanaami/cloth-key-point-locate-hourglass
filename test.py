import csv
from inference import Inference
import cv2


img_path = '/home/jxy/Documents/服饰关键点定位/data/test_b/'
g_test_data_file = '/home/jxy/Documents/服饰关键点定位/data/test_b/test.csv'
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


def testing(model_blouse=None, model_dress=None, model_outwear=None, model_skirt=None, model_trousers=None):
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
	inf_blouse = Inference(config_file='config_blouse.cfg', model=model_blouse, yoloModel=None,
						   nFeat=nFeats,
						   nStack=nStacks,
						   nModules=nModules,
						   nLow=nLow,
						   batch_size=batch_size,
						   drop_rate=dropout_rate,
						   lear_rate=learning_rate,
						   dacay=learning_rate_decay,
						   dacay_step=decay_step,
						   w_loss=weight_loss)

	inf_dress = Inference(config_file='config_dress.cfg', model=model_dress, yoloModel=None,
						  nFeat=nFeats,
						  nStack=nStacks,
						  nModules=nModules,
						  nLow=nLow,
						  batch_size=batch_size,
						  drop_rate=dropout_rate,
						  lear_rate=learning_rate,
						  dacay=learning_rate_decay,
						  dacay_step=decay_step,
						  w_loss=weight_loss)

	inf_outwear = Inference(config_file='config_outwear.cfg', model=model_outwear, yoloModel=None,
							nFeat=nFeats,
							nStack=nStacks,
							nModules=nModules,
							nLow=nLow,
							batch_size=batch_size,
							drop_rate=dropout_rate,
							lear_rate=learning_rate,
							dacay=learning_rate_decay,
							dacay_step=decay_step,
							w_loss=weight_loss)

	inf_skirt = Inference(config_file='config_skirt.cfg', model=model_skirt, yoloModel=None,
						  nFeat=nFeats,
						  nStack=nStacks,
						  nModules=nModules,
						  nLow=nLow,
						  batch_size=batch_size,
						  drop_rate=dropout_rate,
						  lear_rate=learning_rate,
						  dacay=learning_rate_decay,
						  dacay_step=decay_step,
						  w_loss=weight_loss)

	inf_trousers = Inference(config_file='config_trousers.cfg', model=model_trousers, yoloModel=None,
							 nFeat=nFeats,
							 nStack=nStacks,
							 nModules=nModules,
							 nLow=nLow,
							 batch_size=batch_size,
							 drop_rate=dropout_rate,
							 lear_rate=learning_rate,
							 dacay=learning_rate_decay,
							 dacay_step=decay_step,
							 w_loss=weight_loss)

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
	images = []
	for name in imgs:
		img_full_path = img_path + name
		images.append(img_full_path)

	# for i in range(2):  # 先用两张图片测试
	for i in range(len(images)):
		print(str(i + 1) + '/' + str(len(images)))
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
	write_path = '/home/jxy/Documents/服饰关键点定位/data/test_mse_b.csv'
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
	img_path = '/home/jxy/Documents/服饰关键点定位/data/test/Images/blouse/'
	name = '0a979410fa2c9671cc1ba6af410a1d16.jpg'
	name = img_path + name
	inf = Inference(config_file='config_blouse.cfg',
					model='model_blouse_mse_50',
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
	modelblouse   = 'model_blouse_mse_50'
	modeldress    = 'model_dress_mse_50'
	modeloutwear  = 'model_outwear_mse_50'
	modelskirt    = 'model_skirt_mse_50'
	modeltrousers = 'model_trousers_mse_50'

	testing(model_blouse=modelblouse, model_dress=modeldress,	model_outwear=modeloutwear,	model_skirt=modelskirt,	model_trousers=modeltrousers)

	#test_1()
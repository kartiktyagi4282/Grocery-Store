import sys
import tensorflow as tf

class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2


class Model: 
	batchSize = 50
	imgSize = (128, 32)
	maxTextLen = 32

	def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
		self.charList = charList
		self.decoderType = decoderType
		self.mustRestore = mustRestore
		self.snapID = 0

		self.inputImgs = tf.placeholder(tf.float32, shape=(Model.batchSize, Model.imgSize[0], Model.imgSize[1]))
		cnnOut4d = self.setupCNN(self.inputImgs)

		rnnOut3d = self.setupRNN(cnnOut4d)

		(self.loss, self.decoder) = self.setupCTC(rnnOut3d)

		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
		(self.sess, self.saver) = self.setupTF()

			
	def setupCNN(self, cnnIn3d):
		cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)

		kernelVals = [5, 5, 3, 3, 3]
		featureVals = [1, 32, 64, 128, 128, 256]
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		numLayers = len(strideVals)

		pool = cnnIn4d
		for i in range(numLayers):
			kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			relu = tf.nn.relu(conv)
			pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

		return pool


	def setupRNN(self, rnnIn4d):
		rnnIn3d = tf.squeeze(rnnIn4d, axis=[2])
		numHidden = 256
		cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
		((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)									
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		return tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
		

	def setupCTC(self, ctcIn3d):
		ctcIn3dTBC = tf.transpose(ctcIn3d, [1, 0, 2])
		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
		self.seqLen = tf.placeholder(tf.int32, [None])
		loss = tf.nn.ctc_loss(labels=self.gtTexts, inputs=ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True)
		if self.decoderType == DecoderType.BestPath:
			decoder = tf.nn.ctc_greedy_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen)
		elif self.decoderType == DecoderType.BeamSearch:
			decoder = tf.nn.ctc_beam_search_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
		elif self.decoderType == DecoderType.WordBeamSearch:
			word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

			chars = str().join(self.charList)
			wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
			corpus = open('../data/corpus.txt').read()
			decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

		return (tf.reduce_mean(loss), decoder)


	def setupTF(self):
		print('Python: '+sys.version)
		print('Tensorflow: '+tf.__version__)

		sess=tf.Session()

		saver = tf.train.Saver(max_to_keep=1)
		modelDir = '../model/'
		latestSnapshot = tf.train.latest_checkpoint(modelDir)

		if self.mustRestore and not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)

		if latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return (sess,saver)


	def toSparse(self, texts):
		indices = []
		values = []
		shape = [len(texts), 0]

		for (batchElement, text) in enumerate(texts):
			labelStr = [self.charList.index(c) for c in text]
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def decoderOutputToText(self, ctcOutput):
		encodedLabelStrs = [[] for i in range(Model.batchSize)]
		if self.decoderType == DecoderType.WordBeamSearch:
			blank=len(self.charList)
			for b in range(Model.batchSize):
				for label in ctcOutput[b]:
					if label==blank:
						break
					encodedLabelStrs[b].append(label)
		else:
			decoded=ctcOutput[0][0] 
			idxDict = { b : [] for b in range(Model.batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				batchElement = idx2d[0]
				encodedLabelStrs[batchElement].append(label)

		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		sparse = self.toSparse(batch.gtTexts)
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001)
		(_, lossVal) = self.sess.run([self.optimizer, self.loss], { self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * Model.batchSize, self.learningRate : rate} )
		self.batchesTrained += 1
		return lossVal


	def inferBatch(self, batch):
		decoded = self.sess.run(self.decoder, { self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * Model.batchSize } )
		return self.decoderOutputToText(decoded)
	

	def save(self):
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)

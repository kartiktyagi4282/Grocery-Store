from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
import codecs


def testCustomOp(feedMat, corpus, chars, wordChars):
	sess=tf.Session()
	sess.run(tf.global_variables_initializer())

	word_beam_search_module = tf.load_op_library('../cpp/proj/TFWordBeamSearch.so')

	mat=tf.placeholder(tf.float32, shape=feedMat.shape)

	assert(len(chars)+1==mat.shape[2])
	decode=word_beam_search_module.word_beam_search(mat, 25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

	res=sess.run(decode, { mat:feedMat })

	blank=len(chars)
	s=''
	for label in res[0]:
		if label==blank:
			break
		s+=chars[label]
		return (res[0], s)


def loadMat(fn):
	mat=np.genfromtxt(fn, delimiter=';')[:,:-1]
	maxT,_=mat.shape
	
	res=np.zeros(mat.shape)
	for t in range(maxT):
		y=mat[t,:]
		e=np.exp(y)
		s=np.sum(e)
		res[t,:]=e/s

	return np.expand_dims(res,1)


def testMiniExample():
	corpus='a ba'
	chars='ab '
	wordChars='ab'
	mat=np.array([[[0.9, 0.1, 0.0, 0.0]],[[0.0, 0.0, 0.0, 1.0]],[[0.6, 0.4, 0.0, 0.0]]]) # 3 time-steps and 4 characters per time time ("a", "b", " ", blank)
	res=testCustomOp(mat, corpus, chars, wordChars)
	print('')
	print('Mini example:')
	print('Label string: ',res[0])
	print('Char string:', '"'+res[1]+'"')


def testRealExample():
	"real example using a sample from a HTR dataset"
	dataPath='../data/bentham/'
	corpus=codecs.open(dataPath+'corpus.txt', 'r', 'utf8').read()
	chars=codecs.open(dataPath+'chars.txt', 'r', 'utf8').read()
	wordChars=codecs.open(dataPath+'wordChars.txt', 'r', 'utf8').read()
	mat=loadMat(dataPath+'mat_2.csv')
	res=testCustomOp(mat, corpus, chars, wordChars)
	print('')
	print('Real example:')
	print('Label string: ',res[0])
	print('Char string:', '"'+res[1]+'"')


if __name__=='__main__':
	testMiniExample()
	testRealExample()
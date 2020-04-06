from __future__ import print_function
import argparse
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import random
import json
import time
import pickle

# stemmer function removes morphological affixes from words
stemmer = LancasterStemmer()

# load dataset from json file which is given in the folder
with open('intents.json') as json_data:
    intents = json.load(json_data)

context = {}

def create_dataset():

    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    # remove duplicates
    classes = sorted(list(set(classes)))

    # create our training data
    training = []
    output = []
    # create an empty array for our output
    output_empty = [0] * len(classes)

    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)

    # create train and test lists
    train_x = list(training[:,0])
    train_y = list(training[:,1])

    return train_x, train_y,words, classes

# build a simple multilayer neural network
def Build_model(input_var, w, b):

	# layer 1 with simple wx + b type and also added aactivation funtion relu
    layer_1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(input_var,w['w1']),b['b1']))
    
    # layer 2 with simple wx + b type and also added aactivation funtion relu
    layer_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer_1,w['w2']),b['b2']))

    # output layer with simple wx + b type and not added aactivation funtion relu for output
    output = tf.nn.bias_add(tf.matmul(layer_2,w['w3']),b['b3'])

    output_softmax = tf.nn.softmax(output)
    return output, output_softmax

# convert whole data into minibatch
def iterate_minibatches(inputs, targets, batchsize):
    # length of input and target must be the same 
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        # just slicing no shuffling, shuffling already done in dataset function in line(65)
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def classify(sentence, sess, prediction, words, x, classes, ERROR_THRESHOLD = 0.25):
    # generate probabilities from the model
    bow_output = bow(sentence, words)
    # Reshape the input which is compatable NN input shape
    bow_output = np.reshape(bow_output,(-1, bow_output.shape[0]))
    # Run the prediction
    results = sess.run([prediction], feed_dict = {x: bow_output})[0][0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, sess, prediction, words, x, classes, userID='123', show_details=False):
    results = classify(sentence, sess, prediction, words, x, classes)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return random.choice(i['responses'])

            results.pop(0)

# training function 
def model(mode = "train", epochs = 100, learning_rate = 0.001):
    
	print("loading dataset...")
	# Calling create_dataset function
	train_x, train_y, words,classes = create_dataset()

	# initialize input data and target  (x,y)--->(train_x, train_y)
	x = tf.placeholder(tf.float32, [None,len(train_x[0])])
	y = tf.placeholder(tf.int32, [None,len(train_y[0])])

	# batch size must be the number of classes
	batch_size = len(train_y[0])

	# initialize weights with respect to training data
	weights = {
	    'w1' : tf.Variable(tf.random_normal([len(train_x[0]),(len(train_x[0])*3)],stddev = 0.1)),
	    'w2' : tf.Variable(tf.random_normal([len(train_x[0]*3),(len(train_x[0])*2)],stddev = 0.1)),
	    'w3' : tf.Variable(tf.random_normal([len(train_x[0]*2),len(train_y[0])],stddev = 0.1))
	}

	# initialize biases with respect to training data 
	biases = {
	    'b1' : tf.Variable(tf.random_normal([len(train_x[0])*3],stddev = 0.1)),
	    'b2' : tf.Variable(tf.random_normal([len(train_x[0])*2],stddev = 0.1)),
	    'b3' : tf.Variable(tf.random_normal([len(train_y[0])],stddev = 0.1))
	}

	# predict output by created neural network
	prediction, prediction_softmax = Build_model(x, weights, biases)

	# for classification softmax_cross_entropy needed
	error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))
	optm = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(error)
	corr = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
	saver = tf.train.Saver()

	# initialization before running the session
	init = tf.global_variables_initializer()

	# creating tensorflow session
	sess = tf.Session()

	# calling initializer
	sess.run(init)

	if mode == "train":
	    # training process starts from here
	    print("Starting training...")
	    # looping for number of epochs (full batch interation)
	    for epoch in range(epochs):
	       	# initialize training error, accuracy and number of minibatch for training 
	        train_err = 0
	        train_acc = 0
	        train_batches = 0
	        # start time for every epoch
	        start_time = time.time()
	        for batch in iterate_minibatches(train_x, train_y, batch_size):
	            # getting input data as batch
	            inputs, targets = batch
	            # run tensorflow session for training
	            _, err, acc= sess.run([optm, error, accuracy],feed_dict = {x: inputs,y: targets})

	            # adding training error and accuracy of all minibatches per epoch to find total training error and accuracy on main data per intertion(epoch)
	            train_err += err
	            train_acc += acc
	            train_batches += 1
	        
	        # print all process of training on terminal    
	        print("Epoch {} of {} took {:.3f}s".format(
	                epoch + 1, epochs, time.time() - start_time))
	        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
	        print("  training accuracy:\t\t{:.2f} %".format(
	                train_acc / train_batches * 100))

	    # saving train model in same folder where the code is 
	    save_path = saver.save(sess,"model_chatbot.ckpt")
	    print("training has been completed and model has been also saved")
	    
	    # save all of our data structures
	    pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

	if mode == "demo":
	    saver.restore(sess,"./model_chatbot.ckpt")
            print("-------> Welcome to Chat Bot Demo <-------")
            while True:
                # this print is just for space between every Q&A
                print()
                # getting input single question from the user
                input_question = raw_input("Question: ")
                # if user doesn't enter anything so system automatically quit from question answering step
                if input_question == '':
                    break
                # this function gives output answer of asked question
                out = response(input_question, sess, prediction_softmax, words, x, classes)
                print("Answer: ", out)

# main function
def main():
    parser = argparse.ArgumentParser()
    # taking mode option (train or demo)
    parser.add_argument('--mode', choices={'train', 'demo'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()
    # call model function to work by the argument which is given in mode option
    model(mode = args.mode)

if __name__ == "__main__":
    main()

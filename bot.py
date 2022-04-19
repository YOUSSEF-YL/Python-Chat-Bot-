

from tensorflow.python.framework import ops
import nltk
from  nltk.stem.lancaster import  LancasterStemmer
from numpy.lib.function_base import piecewise
streamer = LancasterStemmer()
import numpy 
import  tflearn
import random
import  json
import  pickle


#nltk.download('punkt')

with open("brain.json") as file:
    data = json.load(file)
try:
        
        with open("data.pickle","rb") as f:
            words,labels,training,output =pickle.load(f)
except :        
        words =[]
        labels =[]
        docs_X = []
        docs_Y = []
        training =[]
        output =[]

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds =nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_X.append(wrds)
                docs_Y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [streamer.stem(w.upper()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        out_empty =[0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_X):
            bag = []

            wrds = [streamer.stem(w.lower) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

                out_row = out_empty[:]
                out_row[labels.index(docs_Y[x])] =1

                training.append(bag)
                output.append(out_row)

        training = numpy.array(training)
        output = numpy.array(output)

        with open("data.pickle","wb") as f:
          pickle.dump(( words,labels,training,output),f)

#tensorflow.reset_default_graph()
ops.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net,len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)



try:
    model.load("model.tflearn")
except:    
    model.fit(training,output , n_epoch=1000, batch_size=8,show_metric=True)
    model.save("model.tflearn")


def words_bag(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [streamer.stem(word.upper()) for word in s_words]

    for s in s_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("START TALKING WITH THE BOT (Type Q to quit)")
    while True:
        inp = input("You: ")
        if inp.lower() == "Q":
            break

# link top probiblity num to its word

        results = model.predict([words_bag(inp,words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(tag)

        #print(random.choice(responses))


chat()

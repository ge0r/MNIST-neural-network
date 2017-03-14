import numpy as np
import pyglet
import csv
import time

import input

class TrainerTester(object):
    def __init__(self, net, minibatch_size, learning_rate, epochs_num):
        self.network = net
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.epochs_num = epochs_num
        self.net_layers_num = len(self.network.sizes)

        self.delta_layer = []
        self.input_activations = []
        self.score = 0

        # init delta_layer list and input_activations list
        for image in range(0, minibatch_size):
            temp = []
            for neurons_num in self.network.sizes:
                temp.append(np.zeros((neurons_num, 1)))
            self.delta_layer.append(temp)
            self.input_activations.append(temp)

        self.train_images, self.test_images, self.train_labels, self.test_labels = input.prepare_data()
        self.epoch_array = np.zeros((len(self.train_labels), 1))

        # decode audio file in memory, since sound is going to be played more than once
        self.song = pyglet.media.load("activity_completed.wav", streaming=False)

    def generate_minibatch(self):
        high = len(self.train_labels)
        res = np.random.randint(0, high, self.minibatch_size)
        for index in res:
            self.epoch_array[index] = 1

        return res

    def epoch_end(self):
        # if there are zeros remaining in the epoch array continue the epoch
        if self.epoch_array.prod() != 0:
            return True
        else:
            return False

    def output_error(self, label):
        last_z = self.network.Z[-1]

        # the error is the derivative of the cost function (mean squared error/MSE function for this case)
        error = (self.network.output - label) * self.network.activation_diff(last_z)
        return error

    # backpropagation finds the errors of every layer
    def backpropagation(self, img_id):
        hidden_layers_num = self.net_layers_num - 2
        # loop from last hidden layer to first hidden layer
        for layer in range(hidden_layers_num, 0, -1):
            weights = self.network.weights[layer]
            prime = self.network.activation_diff(self.network.Z[layer])
            self.delta_layer[img_id][layer] = np.dot(np.transpose(weights), self.delta_layer[img_id][layer+1]) * prime

    def update_network(self):
        for layer in range(self.net_layers_num-1, 0, -1):
            sum_w = 0
            sum_b = 0
            for image in range(0, self.minibatch_size):
                sum_w += self.delta_layer[image][layer] * np.transpose(self.input_activations[image][layer-1])
                sum_b += self.delta_layer[image][layer]

            dw = self.learning_rate / self.minibatch_size * sum_w
            db = self.learning_rate / self.minibatch_size * sum_b

            self.network.set_layer_weights(layer-1, dw)
            self.network.set_layer_biases(layer, db)

    def evaluate(self, output, label):
        if np.array_equal(self.smooth(output), label):
            self.score += 1

    def smooth(self, output):
        res = np.zeros((10, 1))
        res[np.argmax(output)] = 1
        return res

    def train(self):
        for epoch in xrange(0, self.epochs_num):
            minibatch_count = 0
            while not self.epoch_end():
                minibatch_indeces = self.generate_minibatch()

                for count, index in enumerate(minibatch_indeces):
                    image = self.train_images[index]
                    label = self.train_labels[index]

                    self.network.initialize_input_layer(image)
                    self.network.feed_forward()

                    self.input_activations[count] = self.network.activation
                    self.delta_layer[count][-1] = self.output_error(label)
                    self.backpropagation(count)

                self.update_network()

                minibatch_count += 1
                # print "epoch_"+str(epoch+1)+", minibatch_"+str(minibatch_count)

            # evaluate in every epoch to get more results.
            score = self.test()
            print "epoch_"+str(epoch+1)+", score_"+str(score)+", h_"+str(self.learning_rate)

            # save the parameters to csv file
            ofile = open('csvs/L'+str(len(self.network.sizes))+'.csv', "a")
            writer = csv.writer(ofile, delimiter=',')
            writer.writerow(["%.3f" % score, self.learning_rate, self.minibatch_size, epoch+1]+self.network.sizes)
            ofile.close()

            # reinitialize epoch array
            self.epoch_array = np.zeros((len(self.train_labels), 1))

        # return the score of the highest epoch
        return score

        # when training and testing is done, play the alert tone
        # self.song.play()

    def test(self):
        self.score = 0
        for image, label in zip(self.test_images, self.test_labels):
            self.network.initialize_input_layer(image)
            output = self.network.feed_forward()
            self.evaluate(output, label)

        score = float(self.score)/len(self.test_labels) * 100
        return score

import trainer
import network

t = trainer.TrainerTester(network.Network([784, 100, 40, 10]), minibatch_size=5, learning_rate=0.6, epochs_num=5)
t.train()
t.test()

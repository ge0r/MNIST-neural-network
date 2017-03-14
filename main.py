import trainer
import network
import time

start = time.time()
high_score = 0
epochs = 5

for h in [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.1, 5, 20]:

    for minibatch_size in [3, 5, 10, 15, 20, 50, 100, 250, 1000]:

        for layer1 in [1, 3, 5, 10, 15, 20, 50, 100, 250]:
            t = trainer.TrainerTester(network.Network([784, layer1, 10]), minibatch_size, h, epochs)
            t.train()

            for layer2 in [1, 3, 5, 10, 15, 20, 50, 100, 250]:
                t2 = trainer.TrainerTester(network.Network([784, layer1, layer2, 10]), minibatch_size, h, epochs)
                t2.train()

end = time.time()
print "hours elapsed: "+str((end - start)/3600)

time.sleep(1)

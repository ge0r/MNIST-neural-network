import trainer
import network
import time

start = time.time()
high_score = 0
epochs = 10
count = 0

for h in [0.5]:

    for minibatch_size in [1]:

        for layer1 in [250]:
            t = trainer.TrainerTester(network.Network([784, layer1, 10]), minibatch_size, h, epochs)
            score = t.train()
            if score > high_score:
                high_score = score

            for layer2 in [17]:
                t2 = trainer.TrainerTester(network.Network([784, layer1, layer2, 10]), minibatch_size, h, epochs)
                score = t2.train()
                if score > high_score:
                    high_score = score

                count += 1
                print "completed "+str(count)+" trainings"
end = time.time()

print "hours elapsed: "+str((end - start)/3600)
print "high_score: "+str(high_score)
time.sleep(1)

import trainer
import network
import time
import csv

start = time.time()
high_score = 0
epochs = 5
count = 0

file1 = open('csvs/threeL.csv', "wb")
writer1 = csv.writer(file1, delimiter=',')

file2 = open('csvs/fourL.csv', "wb")
writer2 = csv.writer(file2, delimiter=',')

for h in [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.1, 5, 20]:

    for minibatch_size in [3, 5, 10, 15, 20, 50, 100, 250, 1000]:

        for layer1 in [1, 3, 5, 10, 15, 20, 50, 100, 250]:
            t = trainer.TrainerTester(network.Network([784, layer1, 10]), minibatch_size, h, epochs)
            score = t.train(writer1)
            if score > high_score:
                high_score = score

            for layer2 in [1, 3, 5, 10, 15, 20, 50, 100, 250]:
                t2 = trainer.TrainerTester(network.Network([784, layer1, layer2, 10]), minibatch_size, h, epochs)
                score = t2.train(writer2)
                if score > high_score:
                    high_score = score

                count += 1
                print "completed "+str(count)+" trainings"

end = time.time()

file1.close()
file2.close()

print "hours elapsed: "+str((end - start)/3600)
print "high_score: "+str(high_score)
time.sleep(1)

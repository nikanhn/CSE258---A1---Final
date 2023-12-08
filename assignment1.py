import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import tensorflow as tf
import statistics
from tensorflow.keras import Model


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

###Task 1: Played Prediction###
userIDs = {}
itemIDs = {}
interactions = []

for u,i,_ in allHours:
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)
    interactions.append((u,i))
                                          
random.shuffle(interactions)
nTrain = int(len(interactions) * 0.9)
nTest = len(interactions) - nTrain
interactionsTrain = interactions[:nTrain]
interactionsTest = interactions[nTrain:]

itemsPerUser = defaultdict(list)
usersPerItem = defaultdict(list)
for u,i in interactionsTrain:
    itemsPerUser[u].append(i)
    usersPerItem[i].append(u)


items = list(itemIDs.keys())
class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.gammaU) +\
                            tf.nn.l2_loss(self.gammaI))
    
    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))

optimizer = tf.keras.optimizers.Adam(0.1)
modelBPR = BPRbatch(5, 0.00001)

def trainingStepBPR(model, interactions):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleJ = [], [], []
        for _ in range(Nsamples):
            u,i, = random.choice(interactions) # positive sample
            j = random.choice(items) # negative sample
            while j in itemsPerUser[u]:
                j = random.choice(items)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleJ.append(itemIDs[j])

        loss = model(sampleU,sampleI,sampleJ)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()

for i in range(150):
    obj = trainingStepBPR(modelBPR, interactions)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))

userSet = set()
gameSet = set()
playedSet = set()

for u,g,d in allHours:
    userSet.add(u)
    gameSet.add(g)
    playedSet.add((u,g))

lUserSet = list(userSet)
lGameSet = list(gameSet)

notPlayed = set()
for u,g,d in hoursTrain:
    #u = random.choice(lUserSet)
    g = random.choice(lGameSet)
    while (u,g) in playedSet or (u,g) in notPlayed:
        g = random.choice(lGameSet)
    notPlayed.add((u,g))

playedValid = set()
for u,g,r in hoursTrain:
    playedValid.add((u,g))
played_games = [(user,game,1) for user,game in playedValid]
not_played_games = [(user,game,0) for user,game in notPlayed]
valid_set = played_games + not_played_games


# predictions = []
# for u,g,_ in valid_set:
#     predictions.append(modelBPR.predict(userIDs[u],itemIDs[g])) 
# preds = [1 if i > 0.55 else 0 for i in predictions]    


played_games = [1] * 9999
not_played_games = [0] * 9999
valid_set = played_games + not_played_games

# correct_predictions = 0
# for i in range(len(valid_set)):
#     if preds[i] == valid_set[i]:
#         correct_predictions += 1
# accuracy = correct_predictions / len(preds)

predictions_file = open("predictions_Played.csv", 'w')
predictions_file.write("userID,gameID,prediction\n")  # header
game_popularity = {g: len(usersPerItem[g]) for g in usersPerItem}

for line in open("pairs_Played.csv"):
    if line.startswith("userID"):
        continue  # skip the header
    user, game = line.strip().split(',')

    # Check if user and game exist in dictionaries
    if user in userIDs and game in itemIDs:
        prediction = modelBPR.predict(userIDs[user], itemIDs[game]).numpy()
        prediction_label = 1 if prediction > 0.55 else 0
        predictions_file.write(f"{user},{game},{prediction_label}\n")
    elif user not in userIDs and game in itemIDs:  
        median_game_popularity = statistics.median(sorted(game_popularity.values(), reverse=True))
        label = 1 if game_popularity.get(game, 0) > median_game_popularity else 0
        predictions_file.write(f"{user},{game},{label}\n") 
    elif user not in userIDs and game not in itemIDs:
        predictions_file.write(f"{user},{game},0\n")

predictions_file.close()

### End of Task 1###

###Task 2: Time Played Prediction###

class WarmUpAndDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps):
        super(WarmUpAndDecaySchedule, self).__init__()
        
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        if step <= self.warmup_steps:
            warmup_lr = initial_learning_rate
        else:
            warmup_lr = self.initial_learning_rate * ((self.total_steps - step) / (self.total_steps - self.warmup_steps))
            
        return warmup_lr


def trainingStep(interactions, model):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleR = [], [], []
        for _ in range(Nsamples):
            u,i,r = random.choice(interactions)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleR.append(r)

        loss_mse = model(sampleU,sampleI,sampleR)
        loss = model.reg() + loss_mse*(1-model.lamb)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy(), loss_mse.numpy(), optimizer.learning_rate.numpy()

def evalStep(interactions, model):
    Nsamples = 50000
    sampleU, sampleI, sampleR = [], [], []
    for _ in range(Nsamples):
        u,i,r = random.choice(interactions)
        sampleU.append(userIDs[u])
        sampleI.append(itemIDs[i])
        sampleR.append(r)

    loss_mse = model(sampleU,sampleI,sampleR)
    loss = model.reg() + loss_mse * (1-model.lamb)
    return loss.numpy(), loss_mse.numpy()

class LatentFactorModel(tf.keras.Model):
    def __init__(self, mu, K, lamb):
        super(LatentFactorModel, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(mu)
        # Initialize to small random values
        # self.betaU = tf.Variable([avg_per_user.get(iid, 0.0) for iid in userIDs])
        # self.betaI = tf.Variable([avg_per_item.get(iid, 0.0) for iid in itemIDs])
        self.betaU = tf.Variable([0.0 for iid in userIDs])
        self.betaI = tf.Variable([0.0 for iid in itemIDs])
        # self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.0000001))
        # self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.0000001))
        self.gammaU = tf.Variable(tf.zeros([len(userIDs), K]))
        self.gammaI = tf.Variable(tf.zeros([len(itemIDs), K]))

        self.lamb = lamb

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
        p = self.alpha + self.betaU[u] + self.betaI[i] + tf.tensordot(self.gammaI[i], self.gammaU[u], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.reduce_sum(self.betaU**2) + tf.reduce_sum(self.betaI**2) + tf.reduce_sum(self.gammaU**2) + tf.reduce_sum(self.gammaI**2))
    
    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        pred = self.alpha + beta_u + beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
            #    tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return pred
    
    # Loss
    def call(self, sampleU, sampleI, sampleR):
        pred = self.predictSample(sampleU, sampleI)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return (2*tf.nn.l2_loss(pred - r)) / len(sampleR)
    

def readJSON(path):
  for l in gzip.open(path, 'rt'):
    d = eval(l)
    u = d['userID']
    try:
      g = d['gameID']
    except Exception as e:
      g = None
    yield u,g,d

def split_dataset(ratio_train, ratio_validation, dataset):
    if ratio_train + ratio_validation > 1:
        raise ValueError("The sum of training and validation ratios should not exceed 1.")

    total_size = len(dataset)
    train_size = int(ratio_train * total_size)

    train_set = dataset[:train_size]
    validation_set = dataset[train_size:]

    return train_set, validation_set

userIDs = {}
itemIDs = {}
interactions = []

avg_per_user = {}
avg_per_item = {}

epoch = 7000
K = 2
lamb = 1e-5
initial_learning_rate = 0.01
warmup_steps = 300

# lr_schedule = WarmUpAndDecaySchedule(initial_learning_rate, warmup_steps, epoch)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

for d in readJSON("train.json.gz"):
    u = d[0]
    i = d[1]
    r = d[2]['hours_transformed']
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)
    interactions.append((u,i,r))

for u, i, r in interactions:
    if u in avg_per_user:
        avg_per_user[u].append(r)
    else:
        avg_per_user[u] = [r]

    if i in avg_per_item:
        avg_per_item[i].append(r)
    else:
        avg_per_item[i] = [r]

avg_per_user = {user: sum(rs)/len(rs) for user, rs in avg_per_user.items()}
avg_per_item = {item: sum(rs)/len(rs) for item, rs in avg_per_item.items()}

# interactions_train, interactions_test = split_dataset(0.8, 0.2, interactions)
interactions_train = interactions
interactions_test = []



# mean rating, just for initialization
mu = sum([r for _,_,r in interactions_train]) / len(interactions_train)

# Experiment with number of factors and regularization rate
model = LatentFactorModel(mu, K, lamb)
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoints', max_to_keep=3000)

train_log_dir = 'logs/train'
eval_log_dir = 'logs/eval'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

for i in range(epoch):
    obj, loss_mse, lr = trainingStep(interactions_train, model)
    print("iteration " + str(i) + ", objective = " + str(obj) + ", lr = " + str(lr), flush=True)
    with train_summary_writer.as_default():
        tf.summary.scalar('train_loss', obj, step=i)
        tf.summary.scalar('train_loss_mse', loss_mse, step=i)
        tf.summary.scalar('learning_rate', lr, step=i)

        for idx, var in enumerate(model.trainable_variables):
            
            name = var.name.replace(':', '_')
            tf.summary.histogram(name + str(idx), var, step=i)
    save_path = manager.save()
    print("Saved checkpoint for epoch {}: {}".format(i, save_path), flush=True)

c = checkpoint.restore("./checkpoints/ckpt-1781")

predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    if u in userIDs and g in itemIDs:
        predictions.write(u + ',' + g + ',' + str(model.predict(userIDs[u], itemIDs[g]).numpy()) + '\n')
    else:
        #print(f'{u} not found')
        predictions.write(u + ',' + g + ',' + str(mu) + '\n')

predictions.close()

### End of Task 2###
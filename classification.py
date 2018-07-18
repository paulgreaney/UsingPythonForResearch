import numpy as np
import random
import scipy.stats as ss
from sklearn import datasets
iris = datasets.load_iris()

def distance(p1, p2):
    """Find the distance between points p1 and p2."""
    return np.sqrt(np.sum(np.power(p2-p1,2)))

p1=np.array([1,1])
p2=np.array([4,4])
#print(distance(p1,p2))

def majority_vote(votes):
    """Calculate number of votes for each candidate"""
    vote_counts={}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1

    winners = []
    max_count = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)

    return random.choice(winners) #choose at random if tied

def majority_vote_short(votes):
    """Return the most commmon element in votes."""
    mode, count = ss.mstats.mode(votes)
    return mode

def find_nearest_neighbours(p, points, k=5):
    """Find k nearest neighbours of point p and return their indices,"""
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    #find k nearest neighbours
    #predict class of p based on majority vote
    ind = find_nearest_neighbours(p, points, k)
    return majority_vote(outcomes[ind])
    
    
def make_prediction_grid(predictors, outcomes, limits, h, k):
    """Classify each point on the prediction grid."""
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype = int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k)

    return(xx, yy, prediction_grid)

seasons = ["spring", "summer", "autumn", "winter"]

print(list(enumerate(seasons)))



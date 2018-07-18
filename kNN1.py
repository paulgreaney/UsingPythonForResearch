import numpy as np
import random

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

votes= [1,2,3,1,2,3,1,2,3,3,3,3]
winner = majority_vote(votes)
print(winner)


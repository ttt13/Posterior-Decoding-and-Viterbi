#!/usr/bin/python3

import sys
import random
import math

import numpy as np
import operator

#####################################################
#####################################################
# Please enter the number of hours you spent on this
# assignment here
num_hours_i_spent_on_this_assignment = 30
#####################################################
#####################################################

#####################################################
#####################################################
# Give one short piece of feedback about the course so far. What
# have you found most interesting? Is there a topic that you had trouble
# understanding? Are there any changes that could improve the value of the
# course to you? (We will anonymize these before reading them.)
# <Your feedback goes here>
'''Lots of errors in the slides. Provided information was nowhere
near what was needed to complete the assignment. This made the length
of the assignment that much longer.
'''
#####################################################
#####################################################



# Outputs a random integer, according to a multinomial
# distribution specified by probs.
def rand_multinomial(probs):
    # Make sure probs sum to 1
    assert(abs(sum(probs) - 1.0) < 1e-5)
    rand = random.random()
    for index, prob in enumerate(probs):
        if rand < prob:
            return index
        else:
            rand -= prob
    return 0

# Outputs a random key, according to a (key,prob)
# iterator. For a probability dictionary
# d = {"A": 0.9, "C": 0.1}
# call using rand_multinomial_iter(d.items())
def rand_multinomial_iter(iterator):
    rand = random.random()
    for key, prob in iterator:
        if rand < prob:
            return key
        else:
            rand -= prob
    return 0


class HMM():

    def __init__(self):
        self.num_states = 2
        self.prior      = np.array([0.5, 0.5])
        self.transition = np.array([[0.999, 0.001], [0.01, 0.99]])
        self.emission   = np.array([{"A": 0.291, "T": 0.291, "C": 0.209, "G": 0.209},
                                    {"A": 0.169, "T": 0.169, "C": 0.331, "G": 0.331}])

    # Generates a sequence of states and characters from
    # the HMM model.
    # - length: Length of output sequence
    def sample(self, length):
        sequence = []
        states = []
        rand = random.random()
        cur_state = rand_multinomial(self.prior)
        for i in range(length):
            states.append(cur_state)
            char = rand_multinomial_iter(self.emission[cur_state].items())
            sequence.append(char)
            cur_state = rand_multinomial(self.transition[cur_state])
        return sequence, states

    # Generates a emission sequence given a sequence of states
    def generate_sequence(self, states):
        sequence = []
        for state in states:
            char = rand_multinomial_iter(self.emission[state].items())
            sequence.append(char)
        return sequence

    # Outputs the most likely sequence of states given an emission sequence
    # - sequence: String with characters [A,C,T,G]
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    def viterbi(self, sequence):
        ###########################################
        # Start your code
        #print("My code here")

        # Initialize M and Prev to be matrices of size T x D
        M = np.zeros( (len(sequence), self.num_states))
        Prev = np.zeros( (len(sequence), self.num_states), dtype = np.int64)

        # Initial state
        M[0][0] = self.prior[0]
        M[0][1] = self.prior[1]

        len_sequence = len(sequence)
        
        for t in range(1,len_sequence):
            # Grab the current letter.
            letter = sequence[t]
            for i in range(0, self.num_states):
                # Use this list to store both probabilities.
                prob = [0,0]
                for j in range(0, self.num_states):
                    # M[t-1][j] * P(i|j) * P(e_t | i)
                    prob[j] = M[t-1][j] * self.transition[j][i] * self.emission[i][letter] 
                # Viterbi takes the max of these probabilities
                M[t][i] = max(prob)
                # Keep track of what we choose
                Prev[t][i] = np.argmax(prob)

        # Generate the sequence
        path = list(sequence)
        path[len_sequence-1] = np.argmax(M[len_sequence-1][:])
        
        for t in range((len_sequence-2), -1,-1):
            path[t] = Prev[t+1][path[t+1]]

        return path

        # End your code
        ###########################################
    def log_sum(self, factors):
        if abs(min(factors)) > abs(max(factors)):
            a = min(factors)
        else:
            a = max(factors)

        total = 0
        for x in factors:
            total += math.exp(x - a)
        return a + math.log(total)

    # - sequence: String with characters [A,C,T,G]
    # return: posterior distribution. shape should be (len(sequence), 2)
    # Please use log_sum() in posterior computations.
    def posterior(self, sequence):
        ###########################################
        # Start your code
        #print("My code here")
        
        ##### Forward #####
        # Initialize forward to be T x D
        forward = np.zeros( (len(sequence), self.num_states))

        # Log initial states, record their emission too.
        forward[0][0] = math.log(self.prior[0]) + math.log(self.emission[0][sequence[0]])
        forward[0][1] = math.log(self.prior[1]) + math.log(self.emission[1][sequence[0]])

        for t in range(1, len(sequence)):
            for i in range(0, self.num_states):
                for j in range(0, self.num_states):
                    # HMM.P(e_t | X_t+1 = i) * HMM.P(X_t+1 = i | X_t = j) * f[t-1, j], the forward equation.
                    product = forward[t-1][j] + math.log(self.transition[j][i]) + math.log(self.emission[i][sequence[t]])
                    # Do not log sum 0 because log(0) = 1. This would throw things off.
                    if forward[t][i] == 0:
                        forward[t][i] = self.log_sum([product])
                    else:
                        forward[t][i] = self.log_sum([forward[t][i], product])

        ##### Backward #####
        # Note that log(1) = 0.
        last_backward_row = math.log(1)
        # Initialize backward to be T x D
        backward = np.zeros( (len(sequence), self.num_states))

        # Moving backwards so initialize end of backward matrix to be [0,0]
        backward[len(sequence)-1][0] = last_backward_row 
        backward[len(sequence)-1][1] = last_backward_row 
        for t in range(len(sequence)-2, -1, -1):
            for i in range(0, self.num_states):
                for j in range(0, self.num_states):
                    # Smoothing: P(e_k+1:t | X_k)
                    product = backward[t+1][j] + math.log(self.transition[i][j]) + math.log(self.emission[j][sequence[t+1]])
                    # Do not log sum 0 because log(0) = 1.
                    if backward[t][i] == 0:
                        backward[t][i] = self.log_sum([product])
                    else:
                        backward[t][i] = self.log_sum([backward[t][i], product])
        
        # Determine alpha for normalization
        alpha = self.log_sum(forward[:,-1])
        alpha = 1/alpha

        # Combine forward and backward and alpha.
        Posterior_probabilities = np.zeros((len(sequence), self.num_states))
        for i in range(0, len(sequence)):
            for j in range(0, self.num_states):
                Posterior_probabilities[i][j] = (alpha * forward[i][j] * backward[i][j])
        
        return Posterior_probabilities

    # End your code
    ###########################################
    # Output the most likely state for each symbol in an emmision sequence
    # - sequence: posterior probabilities received from posterior()
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    def posterior_decode(self, posterior):
        nSamples  = len(sequence)
        post = self.posterior(sequence)
        best_path = np.zeros(nSamples)
        for t in range(nSamples):
            best_path[t], _ = max(enumerate(post[t]), key=operator.itemgetter(1))
        return list(best_path.astype(int))



def read_sequences(filename):
    inputs = []
    with open(filename, "r") as f:
        for line in f:
            inputs.append(line.strip())
    return inputs

def write_sequence(filename, sequence):
    with open(filename, "w") as f:
        f.write("".join(sequence))

def write_output(filename, viterbi, posterior):
    vit_file_name = filename[:-4]+'_viterbi_output.txt' 
    with open(vit_file_name, "a") as f:
        for state in range(2):
            f.write(str(viterbi.count(state)))
            f.write("\n")
        f.write(" ".join(map(str, viterbi)))
        f.write("\n")

    pos_file_name = filename[:-4]+'_posteri_output.txt' 
    with open(pos_file_name, "a") as f:
        for state in range(2):
            f.write(str(posterior.count(state)))
            f.write("\n")
        f.write(" ".join(map(str, posterior)))
        f.write("\n")


if __name__ == '__main__':

    hmm = HMM()

    file = sys.argv[1]
    sequences  = read_sequences(file)
    for sequence in sequences:
        viterbi   = hmm.viterbi(sequence)
        posterior = hmm.posterior_decode(sequence)
        write_output(file, viterbi, posterior) # posterior



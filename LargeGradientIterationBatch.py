#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Large-scale IRL with neural network
'''
import numpy as np
import numpy.random as rn
from itertools import product
import numpy as np
import time
import CUtility
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
class LargeGradientIRL(object):
    def __init__(self, n_actions, n_states, transitionProbability,
                 featureFunction,discount,learning_rate,trajectories,epochs):
        self.n_actions =n_actions
        self.n_states = n_states
        self.discount=discount
        self.transitionProbability=transitionProbability
        self.featureFunction=featureFunction
        self.trajectories=trajectories
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.approxStructure=[self.featureFunction(0).shape[0],100*self.featureFunction(0).shape[0],100*self.featureFunction(0).shape[0],10*self.featureFunction(0).shape[0],1]

    def gradientIterationIRL(self, ground_r):
        print "nn-based large scale gradient iteration irl"
        # Place holder for each batch: for each possible transition, save: nextstateindex, prob, feature
        tf.reset_default_graph()
        net_size=self.approxStructure
        weight_list=[]
        biase_list=[]
        for i_net in range(1,len(net_size)):
            weight =tf.Variable(tf.random_uniform(shape=(net_size[i_net-1], net_size[i_net]) , minval=-np.sqrt(6./(net_size[i_net-1]+net_size[i_net])), maxval=np.sqrt(6./(net_size[i_net-1]+net_size[i_net]))),name="weight"+str(i_net-1))
            biase = tf.Variable(tf.zeros([net_size[i_net]]), name="biases"+str(i_net-1))
            weight_list.append(weight)
            biase_list.append(biase)
        approxValue={}
        def approx(i):
            if(i in approxValue):
                return approxValue[i]
            feature=tf.constant(self.featureFunction(i).reshape(1,self.featureFunction(i).shape[0]),dtype=tf.float32)
            output=[feature]
            for i_net in range(len(net_size)-1):
                weight=weight_list[i_net]
                biase=biase_list[i_net]
                output.append(tf.nn.tanh(tf.matmul(output[-1], weight) + biase))
            approxValue[i]=output[-1][0][0]
            return approxValue[i]
        Q={}
        def QValue(state,action):
            if state in Q:
               return Q[state][action]
            QValue_list=[]
            for j in range(self.n_actions):
                transition=self.transitionProbability(state,j)
                QValue=0
                for nextstate in transition.keys():
                    QValue+=transition[nextstate]*approx(nextstate)
                QValue_list.append(QValue)
            Q[state]=QValue_list
            return Q[state][action]
        V={}
        def Value(state):
            if state in V:
               return V[state]
            QValue_list=[]
            for j in range(self.n_actions):
                QValue_list.append(QValue(state,j))
            V[state]=tf.reduce_max(QValue_list)
            return V[state]
        R={}
        def Reward(state):
            if state in R:
               return R[state]
            R[state]=approx(state)-self.discount*Value(state)
            return R[state]

        # construct the negative loglikelihood function
        negativeloglikelihood=0
        for trajectory in self.trajectories:
            for s,a,r in trajectory:
                QValue_list=[]
                for j in range(self.n_actions):
                    QValue_list.append(QValue(s,j))
                QValue_list=QValue_list-QValue_list[a]
                QValue_list=tf.stack(QValue_list)
                Exp_QValue_list=tf.exp(QValue_list)
                Sum_QValue_list=tf.reduce_sum(Exp_QValue_list)
                negativeloglikelihood+=tf.log(Sum_QValue_list)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(negativeloglikelihood)
        reward=[Reward(i) for i in range(self.n_states)]
        reward=tf.reshape(tf.stack(reward),shape=(self.n_states,))
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        e=0
        result_logcor=[]
        while(True):
            start_time=time.time()
            result=sess.run([optimizer,negativeloglikelihood,reward])
            print time.time()-start_time,e,result[1], np.corrcoef(ground_r,result[2])[0,1]
            result_logcor.append([result[2],np.corrcoef(ground_r,result[2])[0,1]])
            if(result[1]<1 or e>self.epochs):
                break;
            e=e+1 
        return result_logcor


#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Large-scale IRL via Gassian Process
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
class GaussLargeGradientIRL(object):
    def __init__(self, n_actions, n_states, transitionProbability,
                 featureFunction,discount,learning_rate,trajectories,epochs):
        self.n_actions =n_actions
        self.n_states = n_states
        self.discount=discount
        self.transitionProbability=transitionProbability
        self.featureFunction=featureFunction
        self.trajectories=trajectories.reshape((1,-1,3))
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.approxStructure=[self.featureFunction(0).shape[0],self.featureFunction(0).shape[0],self.featureFunction(0).shape[0],self.featureFunction(0).shape[0],1]

    def gradientIterationIRL(self, ground_r):
        print "gp-based large scale gradient iteration irl"
        # Batch based large scale IRL
        tf.reset_default_graph()

        self.support_set=[0,8,16]
        # ARD kernel
        def ARDKernel(f1,f2,para):
            return para[0]*tf.exp(-tf.reduce_sum(tf.multiply(f1-f2,tf.multiply(f1-f2,para[1]))))

        # ARD parameter: para=[beta,Lambda]
        beta=tf.Variable(tf.random_uniform(shape=(1,1)))
        Lambda=tf.Variable(tf.random_uniform(shape=(self.featureFunction(0).shape[0],1)))
        prior_u=tf.Variable(tf.random_uniform(shape=(len(self.support_set),1)))

        # Supporting point: u, S_u
        support_features=[]
        for index in self.support_set:
            support_features.append(tf.constant(self.featureFunction(index),dtype=tf.float32))

        # K_{S_u,S_u}
        K_uuv=[]
        for feature1 in support_features:
            for feature2 in support_features:
                K_uuv.append(ARDKernel(feature1,feature2,[beta,Lambda]))
        K_uu=tf.reshape(tf.stack(K_uuv,axis=0),shape=(len(support_features),len(support_features)))
        
        log_VR_uuv=-0.5*tf.matmul(tf.matmul(tf.transpose(prior_u),tf.matrix_inverse(K_uu)),prior_u)-0.5*tf.log(tf.matrix_determinant(K_uu))-self.featureFunction(0).shape[0]/2*0.945713623

        approxValue={}
        def approx(i):
            if(i in approxValue):
                return approxValue[i]
            feature=tf.constant(self.featureFunction(i).reshape(1,self.featureFunction(i).shape[0]),dtype=tf.float32)
            K_suv=[]
            for feature2 in support_features: 
                K_suv.append(ARDKernel(feature,feature2,[beta,Lambda]))
            K_su=tf.reshape(tf.stack(K_suv,axis=0),shape=(1,-1))
            approxValue[i]=tf.matmul(tf.matmul(K_su,K_uu),prior_u)
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
        negativeloglikelihood+=log_VR_uuv[0][0]
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
            #print time.time()-start_time,e,result[1], np.corrcoef(ground_r,result[2])[0,1]
            result_logcor.append([result[2],np.corrcoef(ground_r,result[2])[0,1]])
            if(result[1]<1 or e>self.epochs):
                break;
            e=e+1 
        return result_logcor


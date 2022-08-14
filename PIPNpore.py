from re import U
#In the name of God
#PIPN for porous media 2D
#Import Libraries
import os
import csv
import linecache
import math
import timeit
from timeit import default_timer as timer
from operator import itemgetter
import numpy as np
from numpy import zeros
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = '12'
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Reshape
#from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Convolution1D, MaxPooling1D
from tensorflow.python.keras.layers import Lambda, concatenate
from tensorflow.python.keras import initializers
from tensorflow import keras
#import h5py


#Global Variables
data = 1 #number of domains
Nd = 2 #dimension of problems, usually 1 or 2
N_boundary = 1 #4*int(np.sqrt(N)) + 4 #number of points on the boundary
num_points = 8727 #N_boundary + N #number of total points
category = 3 #number of variables, e.g., velocity, temperature, pressure
full_list = [] #point number on the whole domain   
BC_list = [] #point number on boundary
interior_list = [] #interior nodes without full, BC, sparse

#Training parameters
J_Loss = 0.00001
LR = 0.0003 #0.0003 #learning rate
Np = 250000 #Number of epochs
Nb = 1 #batch size, note: Nb should be less than data
Ns = 2.0 #scaling the network
pointer = np.zeros(shape=[Nb],dtype=int) #to save indices of batch numbers

#Navier-Stokes
density = 1.0
viscosity = 1.0

#Some functions
def TANHscale(b):
    return tf.tanh(100*b)

def mat_mul(AA, BB):
    return tf.matmul(AA, BB)

def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])

def compute_u(Y):
    return Y[0][:,:,0]

def compute_v(Y):
    return Y[0][:,:,1]

def compute_p(Y):
    return Y[0][:,:,2] 

def compute_dp_dx(X,Y):
    return backend.gradients(Y[0][:,:,2], X)[0][:,:,0]

def compute_dp_dy(X,Y):
    return backend.gradients(Y[0][:,:,2], X)[0][:,:,1]

def map(index):
    return X_train[0][index][0], X_train[0][index][1]

def find(x_i,y_i,data_number,find_value):
    call = -1
    for index in range(num_points):  
        if np.sqrt(np.power(X_train[data_number][index][0]-x_i,2.0) + np.power(X_train[data_number][index][1]-y_i,2.0)) < np.power(10.0,find_value): #np.power(10.0,-4.0):
            call = index
            break
    return call          

def plotCost(Y,name,title):
    plt.plot(Y)
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title(title)
    plt.savefig(name+'.png',dpi = 300,bbox_inches='tight')
    plt.savefig(name+'.eps',bbox_inches='tight')
    plt.clf()
    #plt.show()

def plotGeometry2DPointCloud(X,name,i):   
    x_p = X[i,:,0]
    y_p = X[i,:,1]
    plt.scatter(x_p, y_p)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(name+'.png',dpi=300)
    #plt.savefig(name+'.eps')    
    plt.clf()
    #plt.show()


def plotSolutions2DPointCloud(S,index,title,flag,name):    

    U = np.zeros(num_points,dtype=float)
    if flag==False:    
        for i in range(num_points):
            U[i] = S[index][i] 
    if flag == True:
        U = S 
    x_p = X_train[index,:,0]
    y_p = X_train[index,:,1]
    marker_size= 1.0 
    plt.scatter(x_p/10.0, y_p/10.0, marker_size, U, cmap='jet')
    cbar= plt.colorbar()
    plt.locator_params(axis="x", nbins=6)
    plt.locator_params(axis="y", nbins=6)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(name+'.png',dpi=300)
    plt.savefig(name+'.eps')    
    plt.clf()
    #plt.show()

def plotErrors2DPointCloud(Uexact,Upredict,index,title,name):    

    Up = np.zeros(num_points,dtype=float)
    for i in range(num_points):
        Up[i] = Upredict[index][i] 

    x_p = X_train[index,:,0]
    y_p = X_train[index,:,1]
    marker_size= 1.0
    plt.scatter(x_p/10.0, y_p/10.0, marker_size, np.absolute(Uexact-Up), cmap='jet')
    cbar= plt.colorbar()
    plt.locator_params(axis="x", nbins=6)
    plt.locator_params(axis="y", nbins=6)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(name+'.png',dpi=300)
    plt.savefig(name+'.eps')    
    plt.clf()
    #plt.show()
    
def computeRMSE(Uexact,Upredict,index):

    Up = np.zeros(num_points,dtype=float)
    for i in range(num_points):
        Up[i] = Upredict[index][i] 
    rmse_value = np.sqrt((1.0/num_points)*(np.sum(np.square(Uexact-Up))))
    return rmse_value

def computeRelativeL2(Uexact,Upredict,index):

    Up = np.zeros(num_points,dtype=float)
    for i in range(num_points):
        Up[i] = Upredict[index][i] 
        
    sum1=0
    sum2=0
    for i in range(num_points):
        sum1 += np.square(Up[i]-Uexact[i])
        sum2 += np.square(Uexact[i])

    return np.sqrt(sum1/sum2)

def computeRelativeL2NonWall(Uexact,Upredict,index):

    Up = np.zeros(num_points,dtype=float)
    for i in range(num_points):
        Up[i] = Upredict[index][i] 
        
    sum1=0
    sum2=0
    for i in range(N_boundary,num_points):
        sum1 += np.square(Up[i]-Uexact[i])
        sum2 += np.square(Uexact[i])

    return np.sqrt(sum1/sum2)

#Reading Data 
num_gross = 8727
Gross_train = np.zeros(shape=(data, num_gross, Nd),dtype=float)
num_point_train = np.zeros(shape=(data),dtype=int)
Gross_train_CFD = np.zeros(shape=(data, num_gross, category),dtype=float) #change 4 in the future


x_pore = np.zeros(shape=(num_gross),dtype=float)
y_pore = np.zeros(shape=(num_gross),dtype=float)
u_pore = np.zeros(shape=(num_gross),dtype=float)
v_pore = np.zeros(shape=(num_gross),dtype=float)
p_pore = np.zeros(shape=(num_gross),dtype=float)

def readPorous():
    
    coord = 0

    with open('u.txt', 'r') as f:
        for line in f:
            x_pore[coord] = float(line.split()[0])*0.001/0.001
            y_pore[coord] = float(line.split()[1])*0.001/0.001
            u_pore[coord] = float(line.split()[2])/0.001
            coord += 1
        
    f.close()

    coord = 0
    with open('v.txt', 'r') as f:
        for line in f:
            v_pore[coord] = float(line.split()[2])/0.001
            coord += 1
    f.close()

    coord = 0
    with open('p.txt', 'r') as f:
        for line in f:
            p_pore[coord] = float(line.split()[2])/00.01
            coord += 1
    f.close()

    
    plt.scatter(x_pore,y_pore,s=1.0,c=u_pore, cmap='jet')
    cbar= plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('poreU.png',dpi=300)
    plt.clf()

    plt.scatter(x_pore,y_pore,s=1.0,c=v_pore, cmap='jet')
    cbar= plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('poreV.png',dpi=300)
    plt.clf()

    plt.scatter(x_pore,y_pore,s=1.0,c=p_pore, cmap='jet')
    cbar= plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('poreP.png',dpi=300)
    plt.clf()
    
readPorous()

car_bound = 0
for i in range(len(x_pore)):
    if (np.absolute(u_pore[i]) < np.power(10,-11.0) and np.absolute(v_pore[i]) < np.power(10,-11.0)):
        car_bound += 1

x_bound = np.zeros(shape=(car_bound),dtype=float)
y_bound = np.zeros(shape=(car_bound),dtype=float)
index_bound = np.zeros(shape=(car_bound),dtype=int)

car_bound = 0
for i in range(len(x_pore)):
    if (np.absolute(u_pore[i]) < np.power(10,-11.0) and np.absolute(v_pore[i]) < np.power(10,-11.0)):
        x_bound[car_bound] = x_pore[i]  
        y_bound[car_bound] = y_pore[i]
        index_bound[car_bound] = i
        car_bound += 1

plt.scatter(x_bound,y_bound,s=1.0)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('boundary.png',dpi=300)
plt.clf()

N_boundary = car_bound #We do not consider any boundary points 
num_points = 8727 #memory sensetive

#num_points = capture 

interior_point = num_points - N_boundary
X_train = np.random.normal(size=(data, num_points, Nd))
CFD_train = np.random.normal(size=(data, num_points, category))
X_train_mini = np.random.normal(size=(Nb, num_points, Nd))

for i in range(data):
    for k in range(N_boundary):
        X_train[i][k][0] = x_pore[index_bound[k]]  
        X_train[i][k][1] = y_pore[index_bound[k]] 
        CFD_train[i][k][0] =  u_pore[index_bound[k]]
        CFD_train[i][k][1] =  v_pore[index_bound[k]]
        CFD_train[i][k][2] =  p_pore[index_bound[k]]
    
    index_rest = np.arange(num_points)
    index_rest[~np.isin(index_rest, index_bound)]
    print(len(index_rest))

    for k in range(N_boundary,num_points):
        X_train[i][k][0] = x_pore[index_rest[k-N_boundary]] 
        X_train[i][k][1] = y_pore[index_rest[k-N_boundary]] 
        CFD_train[i][k][0] =  u_pore[index_rest[k-N_boundary]] 
        CFD_train[i][k][1] =  v_pore[index_rest[k-N_boundary]] 
        CFD_train[i][k][2] =  p_pore[index_rest[k-N_boundary]]


#Sparse Observation
k_c = 27
counting = 0
x_pre_sparse = np.random.normal(size=(data, k_c*k_c))
y_pre_sparse = np.random.normal(size=(data, k_c*k_c))
for k in range(data):
    for i in range(k_c):
        for j in range(k_c):
            x_pre_sparse[k][counting] = 100*(i*(0.64/k_c) + 0.01)
            y_pre_sparse[k][counting] = 100*(j*(0.64/k_c) + 0.01)
            counting += 1

plt.scatter(x_pre_sparse[0,:],y_pre_sparse[0,:],s=1.0)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('pre_sparse.png',dpi=300)
plt.clf()

#set_point = np.arange(k_c*k_c)
set_point = []
for i in range(k_c*k_c):
    x_i = x_pre_sparse[0][i]
    y_i = y_pre_sparse[0][i]
    di = np.random.normal(size=(num_points-N_boundary,2))
    for index in range(N_boundary,num_points):
        di[index-N_boundary][0] = 1.0*index  
        di[index-N_boundary][1] = np.sqrt(np.power(X_train[0][index][0]-x_i,2.0) + np.power(X_train[0][index][1]-y_i,2.0))
    di = di[np.argsort(di[:, 1])]
    if di[0][1] < 2.0:
        set_point.append(int(di[0][0]))
    #set_point[i] = int(di[0][0])    
           
sparse_n = len(set_point) #k_c*k_c #counting #200 #num_points
sparse_list = [[-1 for i in range(sparse_n)] for j in range(data)] 
#set_point = np.arange(sparse_n)

print('number of sensors')
print(sparse_n)


def problemSet():

    for i in range(N_boundary):
        BC_list.append(i)

    for i in range(num_points):
        full_list.append(i)
    
    for i in range(data):
        for j in range(sparse_n):
            sparse_list[i][j] = set_point[j]
    
    for i in range(num_points):
        if i in BC_list:
            continue
        interior_list.append(i)

problemSet()

u_sparse = np.random.normal(size=(data, sparse_n))
v_sparse = np.random.normal(size=(data, sparse_n))
p_sparse = np.random.normal(size=(data, sparse_n))
x_sparse = np.random.normal(size=(data, sparse_n))
y_sparse = np.random.normal(size=(data, sparse_n))

for i in range(data):
    for k in range(sparse_n):
        u_sparse[i][k] = CFD_train[i][sparse_list[i][k]][0]
        v_sparse[i][k] = CFD_train[i][sparse_list[i][k]][1]
        p_sparse[i][k] = CFD_train[i][sparse_list[i][k]][2]

        x_sparse[i][k] = X_train[i][sparse_list[i][k]][0]
        y_sparse[i][k] = X_train[i][sparse_list[i][k]][1]

#Plot sparse points
plt.scatter(x_bound,y_bound,s=1.0)
plt.scatter(x_sparse[0,:],y_sparse[0,:],s=1.0)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('sparse.png',dpi=300)
plt.clf()


viscosity = 0.001/0.1 # Pa.s
density = 1.0 # kg/m^3

cfd_u = np.zeros(data*num_points)
cfd_v = np.zeros(data*num_points)
cfd_p = np.zeros(data*num_points)

counter = 0
for j in range(data):
    for i in range(num_points):
        cfd_u[counter] = CFD_train[j][i][0]
        cfd_v[counter] = CFD_train[j][i][1]
        cfd_p[counter] = CFD_train[j][i][2]
        counter += 1

def CFDsolution_u(index):
    return CFD_train[index,:,0]

def CFDsolution_v(index):
    return CFD_train[index,:,1]

def CFDsolution_p(index):
    return CFD_train[index,:,2]

#PointNet
input_points = Input(shape=(num_points, Nd))
g = Convolution1D(int(64*Ns), 1, input_shape=(num_points, Nd), activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(input_points) # I made 3 to 1
#g = BatchNormalization()(g)
g = Convolution1D(int(64*Ns), 1, input_shape=(num_points, Nd), activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(g) #I made 3 to 1 be
#g = BatchNormalization()(g)

seg_part1 = g
g = Convolution1D(int(64*Ns), 1, activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(g)
#g = BatchNormalization()(g)
g = Convolution1D(int(128*Ns), 1, activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(g)
#g = BatchNormalization()(g)
g = Convolution1D(int(1024*Ns), 1, activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(g)
#g = BatchNormalization()(g)

# global_feature
global_feature = MaxPooling1D(pool_size=num_points)(g)
global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)

# point_net_seg
c = concatenate([seg_part1, global_feature])
c = Convolution1D(int(512*Ns), 1, activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(c)
#c = BatchNormalization()(c)
c = Convolution1D(int(256*Ns), 1, activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(c)
#c = BatchNormalization()(c)
c = Convolution1D(int(128*Ns), 1, activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(c)
#c = BatchNormalization()(c)
c = Convolution1D(int(128*Ns), 1, activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(c)
#c = BatchNormalization()(c)
prediction = Convolution1D(category, 1, activation='tanh',kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())(c)
model = Model(inputs=input_points, outputs=prediction)


cost_BC = tf.placeholder(tf.float32, None)
cost_sparse = tf.placeholder(tf.float32, None) 
cost_interior = tf.placeholder(tf.float32, None)

pose_BC = tf.placeholder(tf.int32, None) #Taken from truth
pose_sparse = tf.placeholder(tf.int32, None) #Taken from truth
pose_interior = tf.placeholder(tf.int32, None) #Taken from truth

pose_BC_p = tf.placeholder(tf.int32, None) #Taken from prediction
pose_sparse_p = tf.placeholder(tf.int32, None) #Taken from prediction
pose_interior_p = tf.placeholder(tf.int32, None) #Taken from prediction
    
def ComputeCost_SE(X,Y):

    u_in = tf.gather(tf.reshape(Y[0][:,:,0],[-1]),pose_interior_p)
    v_in = tf.gather(tf.reshape(Y[0][:,:,1],[-1]),pose_interior_p)
    du_dx_in =  tf.gather(tf.reshape(backend.gradients(Y[0][:,:,0], X)[0][:,:,0],[-1]),pose_interior_p) #du/dx in domain
    d2u_dx2_in = tf.gather(tf.reshape(backend.gradients(backend.gradients(Y[0][:,:,0], X)[0][:,:,0], X)[0][:,:,0],[-1]),pose_interior_p) #d2u/dx2 in domain
    du_dy_in =  tf.gather(tf.reshape(backend.gradients(Y[0][:,:,0], X)[0][:,:,1],[-1]),pose_interior_p) #du/dy in domain
    d2u_dy2_in = tf.gather(tf.reshape(backend.gradients(backend.gradients(Y[0][:,:,0], X)[0][:,:,1], X)[0][:,:,1], [-1]),pose_interior_p) #d2u/dy2 in domain
    dv_dx_in =  tf.gather(tf.reshape(backend.gradients(Y[0][:,:,1], X)[0][:,:,0],[-1]),pose_interior_p) #dv/dx in domain
    d2v_dx2_in = tf.gather(tf.reshape(backend.gradients(backend.gradients(Y[0][:,:,1], X)[0][:,:,0], X)[0][:,:,0], [-1]),pose_interior_p) #d2v/dx2 in domain
    dv_dy_in =  tf.gather(tf.reshape(backend.gradients(Y[0][:,:,1], X)[0][:,:,1],[-1]),pose_interior_p) #dv/dy in domain
    d2v_dy2_in = tf.gather(tf.reshape(backend.gradients(backend.gradients(Y[0][:,:,1], X)[0][:,:,1], X)[0][:,:,1], [-1]),pose_interior_p) #d2v/dy2 in domain
    dp_dx_in =  tf.gather(tf.reshape(backend.gradients(Y[0][:,:,2], X)[0][:,:,0],[-1]),pose_interior_p) #dp/dx in domain
    dp_dy_in =  tf.gather(tf.reshape(backend.gradients(Y[0][:,:,2], X)[0][:,:,1],[-1]),pose_interior_p) #dp/dy in domain
    du_dy_in = tf.gather(tf.reshape(backend.gradients(Y[0][:,:,0], X)[0][:,:,1],[-1]),pose_interior_p) #du/dy in domain
    dv_dx_in = tf.gather(tf.reshape(backend.gradients(Y[0][:,:,1], X)[0][:,:,0],[-1]),pose_interior_p) #dv/dx in domain
    
    r1 = 1.0*dp_dx_in - viscosity*(d2u_dx2_in + d2u_dy2_in)
    r2 = 1.0*dp_dy_in - viscosity*(d2v_dx2_in + d2v_dy2_in)
    r3 = du_dx_in + dv_dy_in
   
    u_boundary = tf.gather(tf.reshape(Y[0][:,:,0], [-1]), pose_BC_p) 
    u_sparse = tf.gather(tf.reshape(Y[0][:,:,0], [-1]), pose_sparse_p)
    v_boundary = tf.gather(tf.reshape(Y[0][:,:,1], [-1]), pose_BC_p) 
    v_sparse = tf.gather(tf.reshape(Y[0][:,:,1], [-1]), pose_sparse_p)
    p_boundary = tf.gather(tf.reshape(Y[0][:,:,2], [-1]), pose_BC_p) 
    p_sparse = tf.gather(tf.reshape(Y[0][:,:,2], [-1]), pose_sparse_p)
   
    boundary_u_truth = tf.gather(cfd_u, pose_BC)
    boundary_u_truth = tf.cast(boundary_u_truth, dtype='float32')
    sparse_u_truth = tf.gather(cfd_u, pose_sparse) 
    sparse_u_truth = tf.cast(sparse_u_truth, dtype='float32')

    boundary_v_truth = tf.gather(cfd_v, pose_BC)
    boundary_v_truth = tf.cast(boundary_v_truth, dtype='float32')
    
    sparse_v_truth = tf.gather(cfd_v, pose_sparse) 
    sparse_v_truth = tf.cast(sparse_v_truth, dtype='float32')

    sparse_p_truth = tf.gather(cfd_p, pose_sparse) 
    sparse_p_truth = tf.cast(sparse_p_truth, dtype='float32')

    PDE_cost = tf.reduce_mean(tf.square(r1)+tf.square(r2)+tf.square(r3))
    BC_cost = tf.reduce_mean(tf.square(u_boundary - 0.0)+tf.square(v_boundary - 0.0))
    
    Sparse_cost = tf.reduce_mean(tf.square(u_sparse - sparse_u_truth)+tf.square(v_sparse - sparse_v_truth)+tf.square(p_sparse - sparse_p_truth))

    return (100.0*PDE_cost + 100.0*Sparse_cost + BC_cost)
    #return (PDE_cost + Sparse_cost)
    #return (Sparse_cost)

def build_model_Porous():
    
    LOSS_Total = []
    LOSS_Total_u = []
    min_loss = 1000
    min_relative_u = 1000
    converge_iteration = 0
    criteria = J_Loss

    cost = ComputeCost_SE(model.inputs,model.outputs)    
    vel_u = compute_u(model.outputs)
    vel_v = compute_v(model.outputs)
    vel_p = compute_p(model.outputs)
    vel_dp_dx = compute_dp_dx(model.inputs,model.outputs)
    vel_dp_dy = compute_dp_dy(model.inputs,model.outputs)
    
    u_final = np.zeros((data,num_points),dtype=float)
    v_final = np.zeros((data,num_points),dtype=float)
    p_final = np.zeros((data,num_points),dtype=float)
    dp_dx_final = np.zeros((data,num_points),dtype=float)
    dp_dy_final = np.zeros((data,num_points),dtype=float)

    optimizer = tf.train.AdamOptimizer(learning_rate = LR , beta1=0.9, beta2=0.999, epsilon=0.000001).minimize(loss = cost)
    init = tf.global_variables_initializer()
      
    with tf.Session() as sess:
        sess.run(init)

        start_ite = timer()

        # training loop
        for epoch in range(Np):
        
            temp_cost = 0
            arr = np.arange(data)
            np.random.shuffle(arr)
            for sb in range(int(data/Nb)):
                pointer = arr[int(sb*Nb):int((sb+1)*Nb)]

                group_BC = np.zeros(int(len(pointer)*len(BC_list)), dtype=int)
                group_sparse = np.zeros(int(len(pointer)*sparse_n), dtype=int)
                group_interior = np.zeros(int(len(pointer)*len(interior_list)), dtype=int)

                catch = 0
                for ii in range(len(pointer)):
                    for jj in range(len(BC_list)):
                        group_BC[catch] = int(pointer[ii]*num_points + jj) 
                        catch += 1
                
                catch = 0
                for ii in range(len(pointer)):
                    for jj in range(sparse_n):
                        group_sparse[catch] = sparse_list[pointer[ii]][jj] + pointer[ii]*num_points 
                        catch += 1

                catch = 0
                for ii in range(len(pointer)):
                    for jj in range(len(interior_list)):
                        group_interior[catch] = int(pointer[ii]*num_points + len(BC_list) + jj)
                        catch += 1

                group_BC_p = np.zeros(int(len(pointer)*len(BC_list)), dtype=int)
                group_sparse_p = np.zeros(int(len(pointer)*sparse_n), dtype=int)
                group_interior_p = np.zeros(int(len(pointer)*len(interior_list)), dtype=int)

                catch = 0
                for ii in range(Nb):
                    for jj in range(len(BC_list)):
                        group_BC_p[catch] = int(ii*num_points + jj) 
                        catch += 1
              
                catch = 0
                for ii in range(Nb):
                    for jj in range(sparse_n):
                        group_sparse_p[catch] = sparse_list[pointer[ii]][jj] + ii*num_points 
                        catch += 1

                catch = 0
                for ii in range(Nb):
                    for jj in range(len(interior_list)):
                        group_interior_p[catch] = int(ii*num_points + len(BC_list) + jj)
                        catch += 1

                X_train_mini = np.take(X_train, pointer[:], axis=0)
                
                gr, temp_cost_m, gr1, gr2, gr3, gr4, gr5, gr6 = sess.run([optimizer, cost, pose_BC, pose_sparse, pose_interior, pose_BC_p, pose_sparse_p, pose_interior_p], feed_dict={input_points:X_train_mini, pose_BC:group_BC, pose_sparse:group_sparse, pose_interior:group_interior, pose_BC_p:group_BC_p, pose_sparse_p:group_sparse_p, pose_interior_p:group_interior_p})
                
                if math.isnan(temp_cost_m):
                    print('Nan Value\n')
                    return
                
                temp_cost += temp_cost_m/int(data/Nb)
                 
                u_out = sess.run([vel_u],feed_dict={input_points:X_train})
                u_final = np.power(u_out[0],1.0)                
                relative_u = computeRelativeL2(CFDsolution_u(0),u_final,0)
                
            print(epoch)
            #print(temp_cost)
            print(relative_u)
            LOSS_Total.append(temp_cost)
            LOSS_Total_u.append(relative_u)

            #if temp_cost < min_loss:
            if relative_u < min_relative_u:
                u_out = sess.run([vel_u],feed_dict={input_points:X_train}) 
                v_out = sess.run([vel_v],feed_dict={input_points:X_train}) 
                p_out = sess.run([vel_p],feed_dict={input_points:X_train}) 
                
                dp_dx_out = sess.run([vel_dp_dx],feed_dict={input_points:X_train})
                dp_dy_out = sess.run([vel_dp_dy],feed_dict={input_points:X_train})
        
                u_final = np.power(u_out[0],1.0) 
                v_final = np.power(v_out[0],1.0)
                p_final = np.power(p_out[0],1.0)
                
                dp_dx_final = np.power(dp_dx_out[0],1.0)
                dp_dy_final = np.power(dp_dy_out[0],1.0) 

                min_loss = temp_cost
                min_relative_u = relative_u
                converge_iteration = epoch
            
            if min_loss < criteria:
                break 
        
        end_ite = timer()
        
        plotCost(LOSS_Total,'Total','Total loss')
        plotCost(LOSS_Total_u,'Totalu ','Total loss u')
        
        for index in range(data):

            plotSolutions2DPointCloud(CFDsolution_u(index),index,'u truth '+str(index),True,'u truth')
            plotSolutions2DPointCloud(u_final,index,'Prediction $\it{u}$ (mm/s)',False,'u prediction')
            plotSolutions2DPointCloud(CFDsolution_v(index),index,'v truth '+str(index),True,'v truth')
            plotSolutions2DPointCloud(v_final,index,'Prediction $\it{v}$ (mm/s)',False,'v prediction')
            plotSolutions2DPointCloud(CFDsolution_p(index),index,'p truth '+str(index),True,'p truth')
            plotSolutions2DPointCloud(p_final,index,'Prediction $\it{p}$ (Pa)',False,'p prediction')
            
            plotErrors2DPointCloud(CFDsolution_u(index),u_final,index,'Absolute error '+'$\it{u}$'+' (mm/s)','u error')
            plotErrors2DPointCloud(CFDsolution_v(index),v_final,index,'Absolute error '+'$\it{v}$'+' (mm/s)','v error')
            plotErrors2DPointCloud(CFDsolution_p(index),p_final,index,'Absolute error '+'$\it{p}$'+' (Pa)','p error')
            

        #Error Analysis Based on RMSE
        error_u = [] ;
        error_v = [] ;
        error_p = [] ;
        
        error_u_rel = [] ;
        error_v_rel = [] ;
        error_p_rel = [] ;

        for index in range(data):
            error_u.append(computeRMSE(CFDsolution_u(index),u_final,index))
            error_v.append(computeRMSE(CFDsolution_v(index),v_final,index))
            error_p.append(computeRMSE(CFDsolution_p(index),p_final,index))
            
            #error_u_rel.append(computeRelativeL2(CFDsolution_u(index),u_final,index))
            #error_v_rel.append(computeRelativeL2(CFDsolution_v(index),v_final,index))
            #error_p_rel.append(computeRelativeL2(CFDsolution_p(index),p_final,index))

            error_u_rel.append(computeRelativeL2NonWall(CFDsolution_u(index),u_final,index))
            error_v_rel.append(computeRelativeL2NonWall(CFDsolution_v(index),v_final,index))
            error_p_rel.append(computeRelativeL2NonWall(CFDsolution_p(index),p_final,index))
         
        for index in range(data):
            print('\n')
            print(index)
            print('error_u:')
            print(error_u[index])
            print('error_v:')
            print(error_v[index])            
            print('error_p:')
            print(error_p[index])
            print('error_u_rel:')
            print(error_u_rel[index])
            print('error_v_rel:')
            print(error_v_rel[index])
            print('error_p_rel:')
            print(error_p_rel[index])                                           
            print('\n')     
    
        print('max RMSE u:')
        print(max(error_u))
        print(error_u.index(max(error_u)))
        print('min RMSE u:')
        print(min(error_u))
        print(error_u.index(min(error_u)))

        print('\n')
        
        print('max RMSE v:')
        print(max(error_v))
        print(error_v.index(max(error_v)))
        print('min RMSE v:')
        print(min(error_v))
        print(error_v.index(min(error_v)))
        
        print('\n')
        
        #print('max RMSE p:')
        #print(max(error_p))
        #print(error_p.index(max(error_p)))
        #print('min RMSE p:')
        #print(min(error_p))
        #print(error_p.index(min(error_p)))

        print('\n')
 
        print('max relative u:')
        print(max(error_u_rel))
        print(error_u_rel.index(max(error_u_rel)))
        print('min relative u:')
        print(min(error_u_rel))
        print(error_u_rel.index(min(error_u_rel)))

        print('\n')

        print('max relative v:')
        print(max(error_v_rel))
        print(error_v_rel.index(max(error_v_rel)))
        print('min relative v:')
        print(min(error_v_rel))
        print(error_v_rel.index(min(error_v_rel)))

        print('\n')

        #print('max relative p:')
        #print(max(error_p_rel))
        #print(error_p_rel.index(max(error_p_rel)))
        #print('min relative p:')
        #print(min(error_p_rel))
        #print(error_p_rel.index(min(error_p_rel)))

        print('\n')

        print('average RMSE u:')
        print(sum(error_u)/len(error_u))
        print('\n')
        print('average RMSE v:')
        print(sum(error_v)/len(error_v))
        
        print('\n')
        
        #print('average RMSE p:')
        #print(sum(error_p)/len(error_p))
       
        print('\n')
        
        print('average relative u:')
        print(sum(error_u_rel)/len(error_u_rel))

        print('\n')

        print('average relative v:')
        print(sum(error_v_rel)/len(error_v_rel))

        print('\n')

        #print('average relative p:')
        #print(sum(error_p_rel)/len(error_p_rel))
                
        print('\n')

        print('converge iteration:')
        print(converge_iteration)

        print('\n')

        print('loss value average values:')
        print(min_loss)

        print('\n')

        print('training time (second):')
        print(end_ite - start_ite)
        
        print('min loss of PDE:')
        print(min(LOSS_Total))
        
        print('min loss of PDE iteration:')
        print(LOSS_Total.index(min(LOSS_Total)))

        print('\n')

        x_p = X_train[index,:,0]
        y_p = X_train[index,:,1]

        with open('u_prediction.txt', 'w') as f1:
            for i in range(num_points):
                f1.write(str(x_p[i])+' '+str(y_p[i])+' '+str(u_final[0][i])+'\n')
        f1.close()

        with open('v_prediction.txt', 'w') as f2:
            for i in range(num_points):
                f2.write(str(x_p[i])+' '+str(y_p[i])+' '+str(v_final[0][i])+'\n')
        f2.close()


        with open('p_prediction.txt', 'w') as f3:
            for i in range(num_points):
                f3.write(str(x_p[i])+' '+str(y_p[i])+' '+str(p_final[0][i])+'\n')
        f3.close()

        #plotErrors2DPointCloud(CFDsolution_u(error_u_rel.index(max(error_u_rel))),u_final,error_u_rel.index(max(error_u_rel)),'max error rel u ')
        #plotErrors2DPointCloud(CFDsolution_u(error_u_rel.index(min(error_u_rel))),u_final,error_u_rel.index(min(error_u_rel)),'min error rel u ')
        #plotErrors2DPointCloud(CFDsolution_v(error_v_rel.index(max(error_v_rel))),v_final,error_v_rel.index(max(error_v_rel)),'max error rel v ')
        #plotErrors2DPointCloud(CFDsolution_v(error_v_rel.index(min(error_v_rel))),v_final,error_v_rel.index(min(error_v_rel)),'min error rel v ')
        #plotErrors2DPointCloud(CFDsolution_p(error_p_rel.index(max(error_p_rel))),p_final,error_p_rel.index(max(error_p_rel)),'max error rel p ')
        #plotErrors2DPointCloud(CFDsolution_p(error_p_rel.index(min(error_p_rel))),p_final,error_p_rel.index(min(error_p_rel)),'min error rel p ')
        
build_model_Porous()
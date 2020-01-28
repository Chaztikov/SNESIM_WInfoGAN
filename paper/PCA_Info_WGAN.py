
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy
import gzip
import pickle
import time
from helpers import *
from sklearn.model_selection import train_test_split
from scipy import stats
from tensorflow.examples.tutorials.mnist import input_data

def MaxScaleCenterData(data):
    data = data - data.min()
    data = data / data.max()
    data = (data - 0.5)*2
    return data

batch_size = 128 
iterations = 2001
dim_show_example = 8
img_dim = 64
img_chan = 1
z_size = 2
lcat_size = 6
number_continuous = 2
flag_WGAN =0 
flag_code_dist = 0
flag_trunc_normal = 0
flag_normal=0
flag_uniform=1
set_code_mu = 0
set_code_sigma = 1
set_code_lower = -1
set_code_upper = 1
sample_directory = './figs' 
model_directory = './models'
collecting_directory = './Uniform_bsz_'+str(batch_size)+'_z_'+str(z_size)+'_zk_'+str(lcat_size)+'_zc_'+str(number_continuous)+'_Its_'+str(iterations)+'_64S'
os.system('mkdir ' + collecting_directory)
seed = 77

flag_code_dist = 0
flag_trunc_normal = 1
flag_normal = 0
flag_uniform = 0
constant_zbatch = np.random.uniform(-1.0,1.0,size=[ lcat_size * dim_show_example , z_size]).astype(np.float32) 


np_rng = np.random.RandomState(seed)


load_dir = '/home/charles/Documents/code/Tensorflow/Data/ChannelizedMedia' 
filename = '/OptSnapshots.pkl.gz'
filename = '/reduced_45000_83_snesim_realizations_0_5000.pkl.gz'


with gzip.open(load_dir + filename,'rb') as fp:
    images = pickle.load(fp)
images = images[:,:2025]
h=64; dh=2

X = images[:,:,:,None]
x_train, x_test = train_test_split(X, test_size=0.3)
x_val, x_test = train_test_split(x_test, test_size=0.10)
# scale and shift
x_train = PreprocessImages(x_train[: , : h+dh, : h+dh, 0], dw=h, dh=h, ht=h, wt=h, doreshape=0)
# apply same scaling to test and validation sets
x_test = MaxScaleCenterData(x_test[:,:h,:h])
x_val = MaxScaleCenterData(x_val[:,:h,:h])

#shuffle
x_train = shuffles(x_train,nshuffles=20)

#use PCA
if flag_code_dist==1:
    data_z = np.copy(x_train)
    data_z = np.reshape(data_z,[x_train.shape[0], x_train.shape[1] * x_train.shape[2]])
    pca_z = PCA(n_components=2)
    zy = pca_z.fit_transform(data_z)


def generator(z):
#     go1 = 512
#     go2 = 256
#     go3 = 128
    with tf.variable_scope("generator"):
        nhchan = 256  #depends on number of deconvolutions / convolutions with d2s
        nz_dim = int(img_dim/8.) 
        nout1 = nhchan * nz_dim**2 * img_chan  #4*img_dim**2 #2*2*4*4*256
        go1 = 128 * img_chan
        go2 = 64 * img_chan
        go3 = 32 * img_chan
        go4 = 1 * img_chan #1
        print(img_chan)

        zP = slim.fully_connected(z, nout1 ,normalizer_fn=slim.batch_norm,            
                                  activation_fn=tf.nn.relu,scope='g_project',
                                  weights_initializer=initializer)
        print("zP", zP.get_shape())
        zCon = tf.reshape(zP,[-1, nz_dim, nz_dim, nhchan])
        print(zCon.get_shape())


        gen1 = slim.convolution2d(zCon,                                  
                                  num_outputs=go1,
                                  kernel_size=[3,3],
                                  padding="SAME",
                                  normalizer_fn=slim.batch_norm,
                                  activation_fn=tf.nn.relu,scope='g_conv1',
                                  weights_initializer=initializer)
        gen1 = tf.depth_to_space(gen1,2)
        print(gen1.get_shape())

        gen2 = slim.convolution2d(gen1,
                                  num_outputs=go2,
                                  kernel_size=[3,3],\
                                  padding="SAME",\
                                  normalizer_fn=slim.batch_norm,\
                                  activation_fn=tf.nn.relu,
                                  scope='g_conv2', \
                                  weights_initializer=initializer)
        gen2 = tf.depth_to_space(gen2,2)
        print(gen2.get_shape())

        gen3 = slim.convolution2d(gen2,
                                  num_outputs=go3,
                                  kernel_size=[3,3],\
                                  padding="SAME",
                                  normalizer_fn=slim.batch_norm,\
                                  activation_fn=tf.nn.relu,
                                  scope='g_conv3', 
                                  weights_initializer=initializer)
        gen3 = tf.depth_to_space(gen3,2)
        print("gen3: ", gen3.get_shape())

        g_out = slim.convolution2d(gen3,
                                   num_outputs=go4,
                                   kernel_size=[32,32],
                                   padding="SAME",\
                                   biases_initializer=None,
                                   activation_fn=tf.nn.tanh,\
                                   scope='g_out', 
                                   weights_initializer=initializer)

        print("g_out: ",g_out.get_shape())
        print("End Generator")
    return g_out

def discriminator(bottom, cat_list,conts, reuse=False):

    with tf.variable_scope("discriminator"):
        do1 = 32 * img_chan
        do2 = 64 * img_chan
        do3 = 128 * img_chan
        do4 = 1024 * img_chan
        do5 = 1 * img_chan
        f = 3
        print(bottom.get_shape())
        dis1 = slim.convolution2d(bottom,
                                  do1,
                                  f,
                                  padding="SAME",\
            biases_initializer=None,activation_fn=lrelu,\
            reuse=reuse,scope='d_conv1',weights_initializer=initializer)
        dis1 = tf.space_to_depth(dis1,2)
        print(dis1.get_shape())

        dis2 = slim.convolution2d(dis1,
                                  do2,
                                  f,
                                  padding="SAME",\
            normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
            reuse=reuse,scope='d_conv2', weights_initializer=initializer)
        dis2 = tf.space_to_depth(dis2,2)
        print(dis2.get_shape())

        dis3 = slim.convolution2d(dis2,
                                  do3,
                                  f,
                                  padding="SAME",\
            normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
            reuse=reuse,scope='d_conv3',weights_initializer=initializer)
        dis3 = tf.space_to_depth(dis3,2)
        print(dis3.get_shape())

        dis4 = slim.fully_connected(slim.flatten(dis3),
                                    do4,
                                    activation_fn=lrelu,\
            reuse=reuse,scope='d_fc1', weights_initializer=initializer)
        print("dis4: ", dis4.get_shape())

        if img_chan > 1:
            dis4 = tf.reshape(dis4,[-1, img_dim, img_dim, img_chan])

#        #Remove/Replace in order to use WGAN
        if flag_WGAN==0:
          d_out = slim.fully_connected(dis4,
                                       do5,
                                       activation_fn=None,\
                                        reuse=reuse,scope='d_out', 
                                        weights_initializer=initializer)
        else:
          d_out = dis4
        print("d_out: ", d_out.get_shape())
    
        print("End Discriminator")
        q_a = slim.fully_connected(dis4,128 * img_chan,normalizer_fn=slim.batch_norm,reuse=reuse,scope='q_fc1', weights_initializer=initializer)
        print("q_a",q_a.get_shape())

        ## AUXILIARY NETWORK, mislabeled.
    with tf.variable_scope("variational_dist"):
        q_cat_outs = []
        for idx,var in enumerate(cat_list):
            q_outA = slim.fully_connected(q_a,var,activation_fn=tf.nn.softmax,reuse=reuse,scope='q_out_cat_'+str(idx), weights_initializer=initializer)
            print("q_outA",q_outA.get_shape())
            q_cat_outs.append(q_outA)
            print("q_cat_outs", len(q_cat_outs))
            print("q_cat_outs[0]", q_cat_outs[0].get_shape())

        q_cont_outs = None
        if conts > 0:
            q_cont_outs = slim.fully_connected(q_a,conts,activation_fn=tf.nn.tanh,reuse=reuse,scope='q_out_cont_'+str(conts), weights_initializer=initializer)
        print("q_cont_outs",q_cont_outs.get_shape())

        print("End Variational")

        return d_out,q_cat_outs,q_cont_outs


tf.reset_default_graph()
initializer = tf.truncated_normal_initializer(stddev=0.02)

z_in = tf.placeholder(shape=[None,z_size],dtype=tf.float32) 
real_in = tf.placeholder(shape=[None, img_dim, img_dim, img_chan],dtype=tf.float32) 

#define all categorical and continuous latent variables
categorical_list = [lcat_size]
latent_cat_in = tf.placeholder(shape=[None,len(categorical_list)],dtype=tf.int32)
latent_cat_list = tf.split(latent_cat_in,len(categorical_list),1)
latent_cont_in = tf.placeholder(shape=[None,number_continuous],dtype=tf.float32)
oh_list = []
for idx,var in enumerate(categorical_list):
    latent_oh = tf.one_hot(tf.reshape(latent_cat_list[idx],[-1]),var)
    oh_list.append(latent_oh)
z_lats = oh_list[:]
z_lats.append(z_in)
z_lats.append(latent_cont_in)
z_lat = tf.concat(z_lats,1)


Gz = generator(z_lat) 
Dx,_,_ = discriminator(real_in,categorical_list,number_continuous)
Dg,QgCat,QgCont = discriminator(Gz,categorical_list,number_continuous,reuse=True) 
print("Gz: ", Gz.get_shape())
print("Dx: ", Dx.get_shape())
print("Dg: ", Dg.get_shape())


#modify discriminator: replace logs by identity for GAN
# d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg))
d_loss = -tf.reduce_mean(  Dx) + tf.reduce_mean( Dg )

#modify g_loss = KL Divergence optimizer + reconstruction error
# g_loss = -tf.reduce_mean(tf.log((Dg/(1-Dg)))) 
#modify also for WGAN
# g_loss = -tf.reduce_mean(tf.log((Dg/(1-Dg)))) #+ tf.reduce_mean( tf.square( tf.subtract(Gz , real_in ) )  )
g_loss = -tf.reduce_mean(Dg) 

#WGAN: clip the gradients of the discriminator (does this include variational component?)
clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')]


#Combine losses for each of the categorical variables.
cat_losses = []
for idx,latent_var in enumerate(oh_list):
    cat_loss = -tf.reduce_sum(latent_var*tf.log(QgCat[idx]),axis=1)
    cat_losses.append(cat_loss)
    
#Combine losses for each of the continous variables.
if number_continuous > 0:
    q_cont_loss = tf.reduce_sum(0.5 * tf.square(latent_cont_in - QgCont), axis=1)
else:
    q_cont_loss = tf.constant(0.0)

q_cont_loss = tf.reduce_mean(q_cont_loss)
q_cat_loss = tf.reduce_mean(cat_losses)
q_loss = tf.add(q_cat_loss,q_cont_loss)
print("q_cont_loss: ", q_cont_loss)
print("q_cat_loss: ", q_cat_loss)
print("q_loss: ", q_loss.get_shape())
tvars = tf.trainable_variables()

trainerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.002,beta1=0.5)
trainerQ = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)

d_grads = trainerD.compute_gradients(d_loss,tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')) 
g_grads = trainerG.compute_gradients(g_loss,tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))
q_grads = trainerQ.compute_gradients(q_loss,tvars) 

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)
update_Q = trainerQ.apply_gradients(q_grads)


summary_loss = []

t0 = time.time()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:  
    sess.run(init)
    for i in range(iterations):
        idx_batch = np.arange(i*batch_size,(i+1)*batch_size) % x_train.shape[0]
        
        #sample random variable values from PC distribution
        if flag_code_dist ==1:
            zs = zy[idx_batch , :z_size]
        #sample according to pre-define distribution
        else:
            if flag_trunc_normal==1:
                stats.truncnorm((set_code_lower - set_code_mu) / set_code_sigma, (set_code_upper - set_code_mu) / set_code_sigma, loc=set_code_mu, scale=set_code_sigma)
            elif flag_normal==1:
                zs = np_rng.normal(set_code_mu , set_code_sigma,size=[batch_size,z_size]).astype(np.float32) #Generate a random z batch
            elif flag_uniform==1:
                zs = np_rng.uniform(set_code_lower , set_code_lower , size=[batch_size,z_size]).astype(np.float32) #Generate a random z batch
        #sample cat/conts variable values from distribution
        lcat = np.random.randint(0,lcat_size,[batch_size,len(categorical_list)]) #Generate random c batch
        lcont = np.random.uniform(-1,1,[batch_size,number_continuous]) #
  
# for mnist
#         xs,_ = mnist.train.next_batch(batch_size) #Draw a sample batch from MNIST dataset.
#         xs = (np.reshape(xs,[batch_size,28,28,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
#         xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
#         xs = (np.reshape(xs,[batch_size,h,w,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
        xs = x_train[idx_batch, :, :]
        
        #change dLoss for WGAN
#         _,dLoss = sess.run([update_D,d_loss],feed_dict={z_in:zs,real_in:xs,latent_cat_in:lcat,latent_cont_in:lcont}) #Update the discriminator
        _,dLoss,_ = sess.run([update_D,d_loss,clip_d],feed_dict={z_in:zs,real_in:xs,latent_cat_in:lcat,latent_cont_in:lcont}) #Update the discriminator
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs,real_in:xs, latent_cat_in:lcat,latent_cont_in:lcont}) #Update the generator, twice for good measure.
        _,qLoss,qK,qC = sess.run([update_Q,q_loss,q_cont_loss,q_cat_loss],feed_dict={z_in:zs,latent_cat_in:lcat,latent_cont_in:lcont}) #Update to optimize mutual information.

        #save summary
        summary_loss.append([gLoss, dLoss, qK, qC, qLoss])
        
        if i % 10 == 0:
            tfinal = time.time()
#             print("i: ",i,"t: %.1f"%(tfinal-t0),"G: %.2f "%str(gLoss) + " D: " + str(dLoss) + " Q: " + str([qK,qC]))
                
            print("i: %d t: %.1f G: %.2f D: %.2f Qi%.2f Qc%.2f" % (i, tfinal-t0, gLoss, dLoss, qK, qC) )
            
            
            z_sample = np.random.uniform(-1.0,1.0,size=[lcat_size * dim_show_example , z_size]).astype(np.float32) #Generate another z batch
            lcat_sample = np.reshape(np.array([e for e in range(lcat_size) for _ in range(dim_show_example)]),[lcat_size * dim_show_example,1])
            a = np.reshape(np.array([[(e/4.5 - 1.)] for e in range(lcat_size) for _ in range(dim_show_example)]),[lcat_size , dim_show_example]).T
            b = np.reshape(a,[lcat_size * dim_show_example,1])
            c = np.zeros_like(b)
            lcont_sample = np.hstack([b,c])

            samples = sess.run(Gz,feed_dict={z_in:constant_zbatch,latent_cat_in:lcat_sample,latent_cont_in:lcont_sample}) #Use new z to get sample images from generator.
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            #Save sample generator images for viewing training progress.
            save_images(np.reshape(samples[0:lcat_size*dim_show_example],[lcat_size*dim_show_example, img_dim, img_dim]),[lcat_size,dim_show_example],sample_directory+'/fig'+str(i)+'.png')
                        
            
        if i % 500 == 0 and i != 0:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess,model_directory+'/model-'+str(i)+'.cptk')
            print( "Saved Model")


# In[301]:

def PlotQQData2Axes(plot_data,set_ylims=[0,1],set_ms=0.9,set_alpha=0.5):
    fig, ax = plt.subplots()
    axes = [ax]
    axes[0].plot( plot_data[:,2] ,'.', c='black', alpha=set_alpha, ms=set_ms,label='Q-Categorical')
    axes[0].plot( plot_data[:,3] ,'.', c='blue', alpha=set_alpha, ms=set_ms,label='Q-Continuous')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss (Q-cat, Q-con)')
    axes[0].grid()
    axes[0].set_ylim(set_ylims)
    lgnd1 = axes[0].legend(loc="upper right", numpoints=1, fontsize=10)
    lgnd1.legendHandles[0]._legmarker.set_markersize(6)
    lgnd1.legendHandles[1]._legmarker.set_markersize(6)


def PlotGDQQData2Axes(plot_data,set_ms=0.9,set_alpha=0.5):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    axes = [ax, ax2]
    axes[0].plot( plot_data[:,0] ,'.',c='green', alpha=set_alpha, ms=set_ms, label='Generator')
    axes[0].plot( plot_data[:,1] ,'.', c='red', alpha=set_alpha, ms=set_ms,label='Discriminator')
    axes[1].plot( plot_data[:,2] ,'.', c='black', alpha=set_alpha, ms=set_ms,label='Q-Categorical')
    axes[1].plot( plot_data[:,3] ,'.', c='blue', alpha=set_alpha, ms=set_ms,label='Q-Continuous')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss (D, G)')
    lgnd0 = axes[0].legend(loc="upper right", numpoints=1, fontsize=10)
    lgnd0.legendHandles[0]._legmarker.set_markersize(6)
    lgnd0.legendHandles[1]._legmarker.set_markersize(6)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss (Q-cat, Q-con)')
    axes[1].grid()
    lgnd1 = axes[1].legend(loc="upper left", numpoints=1, fontsize=10)
    lgnd1.legendHandles[0]._legmarker.set_markersize(6)
    lgnd1.legendHandles[1]._legmarker.set_markersize(6)
#     plt.show()


def PlotGDRatioQQData2Axes(plot_data,set_ylims=[-50,100],set_ms=0.9,set_alpha=0.5):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    axes = [ax, ax2]
    axes[0].plot( plot_data[:,0]/(plot_data[:,1]+1e-2) ,'.',c='green', alpha=set_alpha, ms=set_ms, label='G/D')
    axes[1].plot( plot_data[:,2] ,'.', c='black', alpha=set_alpha, ms=set_ms,label='Q-Categorical')
    axes[1].plot( plot_data[:,3] ,'.', c='blue', alpha=set_alpha, ms=set_ms,label='Q-Continuous')
    axes[0].set_ylim(set_ylims)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss Ratio G/D')
    axes[0].grid()
    lgnd0 = axes[0].legend(loc="upper right", numpoints=1, fontsize=10, markerscale = 10)
#     lgnd0.legendHandles._legmarker.set_markersize(6)
    #lgnd0.legendHandles[1]._legmarker.set_markersize(6)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss (Q-cat, Q-con)')
    axes[1].grid()
    lgnd1 = axes[1].legend(loc="upper left", numpoints=1, fontsize=10, markerscale=10)
#     lgnd1.legendHandles[0]._legmarker.set_markersize(6)
#     lgnd1.legendHandles[1]._legmarker.set_markersize(6)
#     plt.show()


def PlotDGRatioQQData2Axes(plot_data,set_ylims=[-50,100],set_ms=0.9,set_alpha=0.5):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    axes = [ax, ax2]
    axes[0].plot( plot_data[:,1]/(plot_data[:,0]+1e-2) ,'.',c='green', alpha=set_alpha, ms=set_ms, label='D/G')
    axes[1].plot( plot_data[:,2] ,'.', c='black', alpha=set_alpha, ms=set_ms,label='Q-Categorical')
    axes[1].plot( plot_data[:,3] ,'.', c='blue', alpha=set_alpha, ms=set_ms,label='Q-Continuous')
    axes[0].set_ylim(set_ylims)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss Ratio D/G')
    axes[0].grid()
    lgnd0 = axes[0].legend(loc="upper right", numpoints=1, fontsize=10, markerscale = 10)

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss (Q-cat, Q-con)')
    axes[1].grid()
    lgnd1 = axes[1].legend(loc="upper left", numpoints=1, fontsize=10, markerscale=10)

#gLoss, dLoss, qK, qC, QLoss
losses = np.array(summary_loss)

plot_data = np.abs(losses)
PlotGDQQData2Axes(plot_data)
plt.savefig(sample_directory+'/lPlotGDQQData2Axes.png')
plt.show()

plot_data = losses
PlotGDQQData2Axes(plot_data)
plt.savefig(sample_directory+'/PlotGDQQData2Axes.png')
plt.show()

plot_data = losses
PlotGDRatioQQData2Axes(plot_data)
plt.savefig(sample_directory+'/PlotGDRatioQQData2Axes.png')
plt.show()

plot_data = losses
PlotGDRatioQQData2Axes(plot_data,set_ylims=[0,10])
plt.savefig(sample_directory+'/PlotGDRatioQQData2Axes_ylim.png')
plt.show()

plot_data = losses
PlotDGRatioQQData2Axes(plot_data,set_ylims=[0,1])
plt.savefig(sample_directory+'/PlotDGRatioQQData2Axes_ylim.png')
plt.show()

plot_data = losses
PlotQQData2Axes(plot_data)
plt.savefig(sample_directory+'/PlotQQData2Axes.png')
plt.show()

plot_data = losses
PlotQQData2Axes(plot_data,set_ylims=[0,1])
plt.savefig(sample_directory+'/PlotQQData2Axes_ylim.png')
plt.show()

np.savetxt(sample_directory+'/losses',losses)



#Load Saved Session and plot continuous changes in realizations for a sequence of categories  

import tensorflow as tf
x_train = MaxScaleCenterData( np.reshape( mnist.train._images, [55000,28,28]) )

sample_directory = './figs' #Directory to save sample images from generator in.
model_directory = './models' #Directory to load trained model from.
dim_show_example = 10
dim_show_example2 = dim_show_example**2
sess = tf.Session()

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:  
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(model_directory)
    saver.restore(sess,ckpt.model_checkpoint_path)

    i=0
    idx_batch = np.arange(i*batch_size,(i+1)*batch_size) % x_train.shape[0]
    if flag_code_dist==1:
        zs = zy[idx_batch , :z_size]
        constant_zbatch = zs[: dim_show_example, :]
    xs = x_train[idx_batch, :, :]
        

    '''
    Generate Plot of fixed code, for sequence of fixed categories and varying 2 continuous variables
    '''
#     z_sample = np.random.uniform(-1.0,1.0, [lcat_size * dim_show_example**2 , z_size]).astype(np.float32) 
    zr = np.random.uniform(-1.0,1.0, [1 , z_size]).astype(np.float32) 
    zr = np.repeat(zr, lcat_size * dim_show_example**2, axis=0)

    zk = np.array([[e] for _ in range(dim_show_example) for _ in range(dim_show_example)for e in range(lcat_size)])

    zc1=zc2 = np.linspace(-1,1,dim_show_example) 
    zc = np.array( [[zc1[i], zc2[j]] for i in range(dim_show_example) for j in range(dim_show_example)] )
    zc = np.repeat( zc, lcat_size,axis=0)
    print("zr: ", zr.shape)
    print("zk: ", zk.shape)
    print("zc: ", zc.shape)
    samples = sess.run(Gz,feed_dict={z_in: zr,
                                     latent_cat_in:zk,
                                     latent_cont_in:zc}) 

dcz = dim_show_example
dcz2 = dcz**2
if not os.path.exists(sample_directory):
    os.makedirs(sample_directory)
    

    
for zkval in range(lcat_size):
    im_array = samples[zkval::lcat_size ,:,:,0]
    print(im_array.shape)
    Block = np.bmat([ [im_array[i+dim_show_example*j] for i in range(dim_show_example)] for j in range(dim_show_example)])
    Block.shape
    img = plt.imshow(Block)
    img.set_cmap('hot')
    plt.axis('off')
    plt.savefig(sample_directory+'/samples_zk_'+str(zkval)+'.png',dpi=1000, bbox_inches='tight')
    plt.show()
    
    with gzip.open(sample_directory+'/samples_zk_'+str(zkval)+'.pkl.gz','w') as fp:
        pickle.dump( samples, fp)

    
# data_y_pca_recon = pca_y.inverse_transform(constant_zbatch)
# data_y_pca_recon = np.reshape(data_y_pca_recon, [data_y_pca_recon.shape[0], img_dim, img_dim, 1])
# samples_pca_recon = data_y_pca_recon
# save_images(np.reshape(samples_pca_recon[0:lcat_size*dim_show_example],[lcat_size*dim_show_example, img_dim, img_dim]),[lcat_size,dim_show_example],sample_directory+'/fig_test_pca_recon'+'.png')
# save_images(np.reshape(xs[0:lcat_size*dim_show_example],[lcat_size*dim_show_example, img_dim, img_dim]),[lcat_size,dim_show_example],sample_directory+'/fig_test_data'+'.png')
# x1 = np.reshape(samples[0:lcat_size*dim_show_example],[lcat_size*dim_show_example, img_dim, img_dim])
# x2 = np.reshape(xs[0:lcat_size*dim_show_example],[lcat_size*dim_show_example, img_dim, img_dim])
# data_samples = np.concatenate((x1,x2))
# save_images(data_samples,[2*lcat_size,dim_show_example],sample_directory+'/fig_test_data_samples'+'.png')

















from sklearn.metrics import pairwise_distances
gen_data = []

for zk in range(lcat_size):
    gen_data.append(samples[zk::lcat_size,:,:].tolist())

n=2500
d=64
data = (np.reshape(samples[zk::lcat_size],[n,d*d])/2.0 + 0.5)
nmodes=100
pca = PCA(n_components = nmodes, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
data_pca = pca.fit_transform( data )
data_recon = pca.inverse_transform(data_pca)

nshow = 10
nshow2 = nshow**2 
images = np.reshape(pca.components_,[nmodes,d,d])
im_array = images[:nshow2,:,:]
Block = np.bmat( [im_array[i] for i in range(nshow)] )
img = plt.imshow(Block)
img.set_cmap('hot')
plt.axis('off')
plt.savefig('GeneratedSnapshots_Modes.png',dpi=100, bbox_inches='tight')
plt.show()

print("Noise Variance: " , pca.noise_variance_)

plt.plot(np.cumsum(pca.explained_variance_) / (pca.noise_variance_ + np.sum(pca.explained_variance_) ))
plt.xlabel('Number of Modes,')
plt.ylabel('Cumulative Explained Variance ')
plt.grid()
plt.savefig('GeneratedSnapshots_CEV.png',dpi=100, bbox_inches='tight')
plt.show()

pcidx1=0
pcidx2=1
nbins = 5

x = data_pca[:,pcidx1]#/pca.explained_variance_[0]
y = data_pca[:,pcidx2]#/pca.explained_variance_[1]
h, x, y, p = plt.hist2d(x, y, bins = nbins)
plt.clf()
plt.imshow(h, origin = "lower", interpolation = "gaussian")
# plt.plot(x,y,'.')
plt.xlabel('PC '+str(pcidx1))
plt.ylabel('PC '+str(pcidx2))
plt.grid()
plt.colorbar()
plt.savefig("GeneratedSnapshots.pdf",dpi=100, bbox_inches='tight')
plt.show()


from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
# use grid search cross-validation to optimize the bandwidth
params = {'bandwidth': np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit( data_pca[:,:nmodes] )
print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_
# sample points from the data
nsamples_kde = 10
new_data = kde.sample( nsamples_kde, random_state=0)
new_data = pca.inverse_transform(new_data)
# grid data
new_data = new_data.reshape((1, nsamples_kde, -1))
real_data = data[:nsamples_kde].reshape((1, nsamples_kde, -1))
Block1 = np.bmat( [ np.reshape(real_data[0,i,:],[d,d]) for i in range(nsamples_kde)] )
Block2 = np.bmat( [ np.reshape(new_data[0,i,:],[d,d]) for i in range(nsamples_kde)] )
Block = np.vstack((Block1,Block2))
img = plt.imshow(Block)
img.set_cmap('hot')
plt.axis('off')
plt.savefig('GeneratedSnapshots_KDESamples'+str(nmodes)+'.png',dpi=100, bbox_inches='tight')
plt.show()




plt.figure()
for nsamples_kde in 2**np.arange(6,14,1): #[100,1000]:#,10000]:
    new_data = kde.sample( nsamples_kde, random_state=0); new_data = pca.inverse_transform(new_data)
    pwd = pairwise_distances(X=new_data, Y=data, metric='euclidean')
#     plt.plot(pwd.min(axis=0),'.',alpha=0.5,ms=2)
    plt.hist(pwd.min(axis=0),alpha=0.8,bins=10,normed=True, label=str(nsamples_kde)+' samples')
    plt.legend()
plt.grid()
plt.xlabel('Minimum Distance (KDE Samples to Real Data)')
plt.savefig('GeneratedSnapshots_KDESamplesPWD'+str(nmodes)+'.png',dpi=100, bbox_inches='tight')
plt.show()
print("Mean Reconstruction Error: ", np.linalg.norm( (data_recon-data).mean(axis=0),2) )

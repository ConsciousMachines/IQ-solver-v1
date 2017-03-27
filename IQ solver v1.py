import glob
import tensorflow as tf
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imsave

directory = 'IQ stuff/iq/' # path to the puzzle files


box_width = 98
box_height = 76
from_center_height_up = 38
from_center_height_bot = 38
from_center_width = 49

# candidate answer position corners
answers_dict = [
[14, 402],[164, 402],[314, 402],[464, 402],
[14, 506],[164, 506],[314, 506],[464, 508]]


# center positions
center_dict = [
[150, 85],[300, 85],[446, 85],
[150, 194],[300, 194],[446, 194],
[150, 296],[300, 296]]

whole_set = []
whole_set_answers = []
for thing in glob.glob(directory + '*.png'):
    print(thing)
    thing0 = im.open(thing)
    thing1 = np.asarray(thing0, dtype='float32')
    thing2 = np.sum([ thing1[:,:,i] for i in range(0,3)],axis=0)/3
    answers = []
    elements = []
    for i in range(8):
        coords = np.array(answers_dict[i])
        start_corner = coords
        end_corner = coords + np.array([box_width, box_height])
        answer = thing2[ start_corner[1]:end_corner[1], start_corner[0]:end_corner[0]]
        answer = np.array(answer, dtype = 'uint8')
        
        threshold = 140 # 70 works, 150 good for fatter lines. stick to 140 for now
        answer[answer < threshold] = 1
        answer[answer >= threshold] = 0
        answers.append(answer)
    for i in range(8):
        coords = np.array(center_dict[i])
        start_corner = coords - np.array([from_center_width, from_center_height_up])
        end_corner = coords + np.array([from_center_width, from_center_height_bot])
        element = thing2[ start_corner[1]:end_corner[1], start_corner[0]:end_corner[0]]
        element = np.array(element, dtype = 'uint8')

        element[element < threshold] = 1
        element[element >= threshold] = 0
        
        #b = im.fromarray(element*250)
        #b.show()
        elements.append(element)
    whole_set.append(elements)
    whole_set_answers.append(answers)
whole_set = np.array(whole_set)
whole_set_answers = np.array(whole_set_answers)
print(whole_set.shape)
print(whole_set_answers.shape)



data_size = whole_set.shape[0]
train_size = 0.9
train_size_index = int(train_size*data_size)
train_data = np.array( whole_set[ :train_size_index,:,:,:] )
test_data = np.array( whole_set[ train_size_index:,:,:,:] )



    

def thing(i):
    c_idx = i
    thing= im.open(directory+'0_'+str(i)+'.png')
    thing.show()
    ex1 = np.squeeze( train_data[c_idx,:,:,:] )
    print(ex1.shape)

    ex1b = ex1[[0,1,3,4,6],:,:] # input to go into the function
    print(ex1b.shape)

    ex1c = ex1[[1,2,4,5,7],:,:] # output to train per example

    ex1ans = np.squeeze( whole_set_answers[c_idx] ) # answers to compare after training


    x_placeholder = tf.placeholder(tf.float32,[1, box_height, box_width,1])
    y_placeholder = tf.placeholder(tf.float32,[1, box_height, box_width,1])
    x_in = tf.unstack(x_placeholder)
    y_input = tf.unstack(y_placeholder)

    n1 = 3
    n2 = 3
    n3 = 3
    n4 = 3
    n5 = 3
    n6 = 1

    conv_filter1 = tf.Variable(np.random.rand(3, 3, 1, n1), dtype = tf.float32)
    conv_filter2 = tf.Variable(np.random.rand(3, 3, n1, n2), dtype = tf.float32)
    conv_filter3 = tf.Variable(np.random.rand(3, 3, n2, n3), dtype = tf.float32)
    conv_filter4 = tf.Variable(np.random.rand(3, 3, n3, n4), dtype = tf.float32)
    conv_filter5 = tf.Variable(np.random.rand(3, 3, n4, 1), dtype = tf.float32)
    conv_filter5a = tf.Variable(np.random.rand(3, 3, n4, n5), dtype = tf.float32)
    conv_filter6 = tf.Variable(np.random.rand(3, 3, n5, n6), dtype = tf.float32)

    b1 = tf.Variable(np.random.rand(n1), dtype = tf.float32)
    b2 = tf.Variable(np.random.rand(n2), dtype = tf.float32)
    b3 = tf.Variable(np.random.rand(n3), dtype = tf.float32)
    b4 = tf.Variable(np.random.rand(n4), dtype = tf.float32)
    b5 = tf.Variable(np.random.rand(1), dtype = tf.float32)
    b5a = tf.Variable(np.random.rand(n5), dtype = tf.float32)
    b6 = tf.Variable(np.random.rand(n6), dtype = tf.float32)


    def nn_y(inp):
        x_in = inp
        padding1 = 'SAME' #'VALID'
        strides1 = [1,1,1,1]

        l1 = tf.nn.conv2d( x_in, conv_filter1, strides1,padding=padding1)
        l1 = tf.nn.relu( tf.nn.bias_add( l1, b1 ))
        l1 = tf.nn.max_pool( l1 ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')
        

        l2 = tf.nn.conv2d( l1, conv_filter2, strides1,padding=padding1)
        l2 = tf.nn.relu( tf.nn.bias_add( l2, b2 ))
        l2 = tf.nn.max_pool( l2 ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')

        l3 = tf.nn.conv2d( l2, conv_filter3, strides1,padding=padding1)
        l3 = tf.nn.relu( tf.nn.bias_add( l3, b3 ))
        l3 = tf.nn.max_pool( l3 ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')
        
        l4 = tf.nn.conv2d( l3, conv_filter4, strides1,padding=padding1)
        l4 = tf.nn.relu( tf.nn.bias_add( l4, b4 ))
        
        l5 = tf.nn.conv2d( l4, conv_filter5, strides1,padding=padding1)
        l5 = tf.nn.relu( tf.nn.bias_add( l5, b5 ))
        '''
        l6 = tf.nn.conv2d( l5, conv_filter6, strides1,padding=padding1)
        l6 = tf.nn.relu( tf.nn.bias_add( l6, b6 ))
        '''
        return l3

    def nn_x(inp):
        x_in = inp
        padding1 = 'SAME' #'VALID'
        strides1 = [1,1,1,1]

        l1 = tf.nn.conv2d( x_in, conv_filter1, strides1,padding=padding1)
        l1 = tf.nn.relu( tf.nn.bias_add( l1, b1 ))
        l1 = tf.nn.max_pool( l1 ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')

        l2 = tf.nn.conv2d( l1, conv_filter2, strides1,padding=padding1)
        l2 = tf.nn.relu( tf.nn.bias_add( l2, b2 ))
        l2 = tf.nn.max_pool( l2 ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')
        
        l3 = tf.nn.conv2d( l2, conv_filter3, strides1,padding=padding1)
        l3 = tf.nn.relu( tf.nn.bias_add( l3, b3 ))
        l3 = tf.nn.max_pool( l3 ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')

        l4 = tf.nn.conv2d( l3, conv_filter4, strides1,padding=padding1)
        l4 = tf.nn.relu( tf.nn.bias_add( l4, b4 ))

        l5 = tf.nn.conv2d( l4, conv_filter5a, strides1,padding=padding1)
        l5 = tf.nn.relu( tf.nn.bias_add( l5, b5a ))

        l6 = tf.nn.conv2d( l5, conv_filter6, strides1,padding=padding1)
        l6 = tf.nn.relu( tf.nn.bias_add( l6, b6 ))
        return l5

    l5 = nn_x(x_in)

    y_input = tf.nn.max_pool( y_input ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')
    y_input = tf.nn.max_pool( y_input ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')
    y_input = tf.nn.max_pool( y_input ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')


    loss = tf.reduce_sum( tf.square( y_input - l5 ))
  

    #train_step = tf.train.AdamOptimizer(0.1).minimize(loss) # 0.1 works
    #train_step = tf.train.RMSPropOptimizer(0.1).minimize(loss)
    train_step = tf.train.AdagradOptimizer(0.2).minimize(loss) # 0.1 works
    epochs = 200
    with tf.Session() as sess:       
        loss_list = []
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            if True:
                for k in range(5):
                    situation = 'train'
                    xxx = np.expand_dims([ex1b[k]],axis=3)
                    yyy = np.expand_dims([ex1c[k]],axis=3)
                    _train_step ,_loss= sess.run([ train_step, loss], 
                        feed_dict={ x_placeholder:xxx,
                                    y_placeholder:yyy})
                    loss_list.append(_loss)

            if e%200==0:
                situation = 'test'
                answer_losses = []
                for i in range(8):
                    ans_i = ex1ans[i]
                    xx =  np.expand_dims([ex1c[4]],axis=3)
                    yy = np.expand_dims( [ans_i],axis=3)
                    _loss2 = sess.run([ loss2], 
                        feed_dict={ x_placeholder:xx,
                                    y_placeholder:yy})
                    answer_losses.append( _loss2 )
                #print(_loss2, 'loss 2')
                ans_idx = np.where( answer_losses == np.min(answer_losses))[0]
                print(answer_losses,'losses')
                print(ans_idx,'ans idx')
                answerr = np.squeeze(255*ex1ans[ans_idx[0]] )
                print(answerr.shape,'answerr shape')
                plt.imshow( answerr)
                plt.savefig(directory2+'xor_graph'+str(i//50))
                plt.draw()
                plt.pause(0.0001)



for i in [0,1,2,4]:
    thing(i)





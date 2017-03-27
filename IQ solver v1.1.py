import glob
import tensorflow as tf
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imsave

# THIS ONE REMOVES COMMON PIXELS OF NO CHANGE AND COMPUTES XOR NN PER PIXEL

directory = '/IQ stuff/iq/'

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

#(42, 8, 76, 98)
#(42, 8, 76, 98)





def thing(i):
    c_idx = i
    thing= im.open(directory+'0_'+str(i)+'.png')
    thing.show()
    ex1 = np.squeeze( whole_set[c_idx,:,:,:] )
    ex1 = np.reshape(ex1,[8,76*98])
    

    ex1ans = np.squeeze( whole_set_answers[c_idx,:,:,:] ) # answers to compare after training
    ex1ans =  np.reshape(ex1ans,[8,76*98])


    print(ex1ans.shape,'ayyy')

    
    check1 = ex1[0,:] + ex1[1,:] + ex1[2,:]
    check2 = ex1[3,:] + ex1[4,:] + ex1[5,:]
    check3 = ex1[6,:] + ex1[7,:] + np.sum(ex1ans,axis=0)
    print(len(check3), 'len check 3')

    x0 = np.squeeze(ex1[0,[np.where(check1 != 0 )]] ) 
    x1 = np.squeeze(ex1[1,[np.where(check1 != 0 )]] )
    y2 = np.squeeze(ex1[2,[np.where(check1 != 0 )]] )
    #print(y2[0,:10])
    print(y2.shape)

    x3 = np.squeeze(ex1[3,[np.where(check2 != 0 )]])
    x4 = np.squeeze(ex1[4,[np.where(check2 != 0 )]])
    y5 = np.squeeze(ex1[5,[np.where(check2 != 0 )]])
    print(x3.shape, 'x3 shape')

    x6 = np.squeeze(ex1[6,[np.where(check3 != 0 )]])
    x7 = np.squeeze(ex1[7,[np.where(check3 != 0 )]])
    y8 = np.squeeze(ex1ans[:,[np.where(check3 != 0 )]])
    print(y8.shape, 'y8 shape')

    ins = [x0,x1,x3,x4,x6,x7]
    outs = [y2,y5]

    

    x1_in = tf.placeholder(tf.float32,[1, None])
    x2_in = tf.placeholder(tf.float32,[1, None])
    y_in = tf.placeholder(tf.float32,[1, None])

    print(type(x1_in),'look')



    n1 = 2
    n2 = 1

    w1 = tf.Variable(np.random.rand(2,n1), dtype = tf.float32)
    w2 = tf.Variable(np.random.rand(n1,n2), dtype = tf.float32) 

    b1 = tf.Variable(np.random.rand(n1), dtype = tf.float32)
    b2 = tf.Variable(np.random.rand(n2), dtype = tf.float32)



    def pool_y(inp): # E X P E R I M E N T A L 
        l1 = tf.nn.max_pool( inp ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')
        #l1 = tf.nn.max_pool( l1 ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')
        #l1 = tf.nn.max_pool( l1 ,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')

        print(l1.get_shape(),'y shape after just pool')
        return l1


    def xor(x1, x2):

        xs = tf.concat([x1,x2],0)
        print(xs.get_shape())

        xs = tf.transpose(xs)


        l1 = tf.matmul(xs, w1)
        l1 = tf.nn.sigmoid( tf.nn.bias_add( l1, b1 ))

        l2 = tf.matmul(l1, w2)
        l2 = tf.nn.sigmoid( tf.nn.bias_add( l2, b2 ))

        return tf.transpose(l2)

    '''

    x1_in = pool_y(x1_in)
    x2_in = pool_y(x2_in)
    y_in = pool_y(y_in)
    '''

    x_3 = xor(x1_in,x2_in)

    loss = tf.reduce_sum( tf.square( y_in - x_3 ))
    train_step = tf.train.AdagradOptimizer(0.5).minimize(loss) # 0.1 works
    epochs = 5000
    with tf.Session() as sess:       
        loss_list = []
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            if True:
                for k in range(2):
                    if k == 0:
                        _train_step ,_loss= sess.run([ train_step, loss], 
                            feed_dict={ x1_in:np.asarray([x0],dtype='float32'),
                                        x2_in:np.array([x1],dtype='float32'),
                                        y_in:np.array([y2],dtype='float32')})
                    if k == 1:
                        _train_step ,_loss= sess.run([ train_step, loss], 
                            feed_dict={ x1_in:np.asarray([x3],dtype='float32'),
                                        x2_in:np.array([x4],dtype='float32'),
                                        y_in:np.array([y5],dtype='float32')})
                    loss_list.append(_loss)
                    
            if e%200==0:
                answer_losses = []
                for i in range(8):
                    xx1 = np.array([x6],dtype='float32')
                    xx2 = np.array([x7],dtype='float32')
                    yy = np.array([y8[i,:]],dtype='float32')
                    _loss = sess.run([ loss ], 
                        feed_dict={ x1_in:xx1,
                                    x2_in:xx2,
                                    y_in:yy})
                    answer_losses.append( _loss )
                ans_idx = np.where( answer_losses == np.min(answer_losses))[0]
                print(answer_losses,'losses')
                answerr = np.squeeze(whole_set_answers[c_idx,ans_idx,:,:])
                plt.imshow( answerr)
                plt.draw()
                plt.pause(0.1)





thing(1)

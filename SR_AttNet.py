from keras.layers import *
from keras import Model


act1 =  LeakyReLU(0.1)
act2= 'relu'
act3= 'sigmoid'

def conv_op(x , f, name):       #  ‘Attention Embedded Convolution Operation’ (AECO)
        flop=0
        flop += 3*3*(x.shape[1]*x.shape[2]*f)
        x1 = Conv2D(f, (3, 3), activation=act1, padding="same", name= name+'_s1')(x)  # s1

        #SR-Attention
        flop += 3*3*(x1.shape[1]*x1.shape[2]*(f//2))
        x1 = Conv2DTranspose(f//2, (3, 3), strides=(2, 2), padding="same", activation=act1, name= name+'_s2')(x1)
        flop += 3*3*(x1.shape[1]*x1.shape[2]*(f//2))
        x1 = Conv2D(f//2, (3, 3), activation=act1, padding="same", strides=(2, 2), name= name+'_s3')(x1)
        flop += 3*3*(x1.shape[1]*x1.shape[2]*(f//2))
        x1 = Conv2DTranspose(f//2, (3, 3), strides=(2, 2), padding="same", activation=act1, name= name+'_s4')(x1)
        flop += 3*3*(x1.shape[1]*x1.shape[2]*(f))
        x1 = Conv2D(f, (3, 3), activation=act3, strides=(2, 2), padding="same", name= name+'_m')(x1)  # m
        #SR-Attention

        # d=1
        flop += 3*3*(x.shape[1]*x.shape[2]*(f))
        conv3 = Conv2D(f, (3, 3), activation=act1, padding="same")(x)
        flop += 3*3*(conv3.shape[1]*conv3.shape[2]*(f))
        conv3 = Conv2D(f, (3, 3), activation=act1, padding="same", name= name+'_preatt')(conv3)

        x3 = multiply([x1,conv3], name= name+'_atted')
        x3 = add([x3, conv3], name= name+'_ym')     # ym
        
        x3 = LayerNormalization([1,2])(x3)
        flop += 1*1*(x3.shape[1]*x3.shape[2]*(x3.shape[3]))
        x3 = DepthwiseConv2D((1,1), activation=act1, padding="same", name= name+'_f1')(x3) # f1

        # d=2
        flop += 3*3*(x.shape[1]*x.shape[2]*(f))
        conv5 = Conv2D(f, (3, 3), activation=act1, padding="same", dilation_rate=(2,2))(x)
        flop += 3*3*(conv5.shape[1]*conv5.shape[2]*(f))
        conv5 = Conv2D(f, (3, 3), activation=act1, padding="same", dilation_rate=(2,2))(conv5)

        x5 = multiply([x1,conv5])
        x5 = add([x5, conv5], name= name+'_zm')     # zm
        
        x5 = LayerNormalization([1,2])(x5)
        flop += 1*1*(x5.shape[1]*x5.shape[2]*(x5.shape[3]))
        x5 = DepthwiseConv2D((1,1), activation=act1, padding="same", name= name+'_f2')(x5) # f2
        return x3, x5, flop


def SR_AttNet(img_dims=256, start_neurons=32):

    input_layer = Input((img_dims, img_dims, 3))

    flops = 0

    # Encoder
    conv1a_3, conv1a_5, flop = conv_op(input_layer, start_neurons * 1, name='conv_1a')
    flops+= flop
    conv1_3, conv1_5, flop = conv_op(conv1a_3, start_neurons * 1, name='conv_1b')    
    flops+= flop
    pool1 = MaxPooling2D((2, 2))(conv1_3)
    pool1 = Dropout(0.25)(pool1)


    conv2a_3, conv2a_5, flop = conv_op(pool1, start_neurons * 2, name='conv_2a')
    flops+= flop
    conv2_3, conv2_5, flop = conv_op(conv2a_3, start_neurons * 2, name='conv_2b')    
    flops+= flop
    pool2 = MaxPooling2D((2, 2))(conv2_3)
    pool2 = Dropout(0.5)(pool2)


    conv3a_3, conv3a_5, flop = conv_op(pool2, start_neurons * 4, name='conv_3a')
    flops+= flop
    conv3_3, conv3_5, flop = conv_op(conv3a_3, start_neurons * 4, name='conv_3b')  
    flops+= flop
    pool3 = MaxPooling2D((2, 2))(conv3_3)
    pool3 = Dropout(0.5)(pool3)

    conv4a_3, conv4a_5, flop = conv_op(pool3, start_neurons * 8, name='conv_4a')
    flops+= flop
    conv4_3, conv4_5, flop = conv_op(conv4a_3, start_neurons * 8, name='conv_4b')  
    flops+= flop
    pool4 = MaxPooling2D((2, 2))(conv4_3)
    pool4 = Dropout(0.5)(pool4)


    # Middle
    flops+= 3*3*pool4.shape[1]*pool4.shape[2]*start_neurons * 16
    convm = Conv2D(start_neurons * 16, (3, 3), activation=act1, padding="same", name='conv_5a')(pool4)
    flops+= 3*3*convm.shape[1]*convm.shape[2]*start_neurons * 16
    convm = Conv2D(start_neurons * 16, (3, 3), activation=act1, padding="same", name='conv_5b')(convm)
    

    # Decoder
    flops+= 3*3*convm.shape[1]*convm.shape[2]*start_neurons * 8
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4_3, conv4a_3])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4a_3, uconv4a_5, flop = conv_op(uconv4 , start_neurons * 8, name='deconv_4a')
    flops+= flop
    uconv4a_3 = concatenate([uconv4a_3, conv4_5, conv4a_5])
    uconv4_3, uconv4_5, flop = conv_op(uconv4a_3 , start_neurons * 8, name='deconv_4b')
    flops+= flop

    flops+= 3*3*uconv4_3.shape[1]*uconv4_3.shape[2]*start_neurons * 4
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4_3)
    uconv3 = concatenate([deconv3, conv3_3, conv3a_3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3a_3, uconv3a_5, flop = conv_op(uconv3 , start_neurons * 4, name='deconv_3a')
    flops+= flop
    uconv3a_3 = concatenate([uconv3a_3, conv3_5, conv3a_5])
    uconv3_3, uconv3_5, flop = conv_op(uconv3a_3 , start_neurons * 4, name='deconv_3b')
    flops+= flop

    flops+= 3*3*uconv3_3.shape[1]*uconv3_3.shape[2]*start_neurons * 2
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3_3)
    uconv2 = concatenate([deconv2, conv2_3, conv2a_3])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2a_3, uconv2a_5, flop = conv_op(uconv2 , start_neurons * 2, name='deconv_2a')
    flops+= flop
    uconv2a_3 = concatenate([uconv2a_3, conv2_5, conv2a_5])
    uconv2_3, uconv2_5, flop = conv_op(uconv2a_3 , start_neurons * 2, name='deconv_2b')
    flops+= flop

    flops+= 3*3*uconv2_3.shape[1]*uconv2_3.shape[2]*start_neurons * 1
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2_3)
    uconv1 = concatenate([deconv1, conv1_3, conv1a_3])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1a_3, uconv1a_5, flop = conv_op(uconv1 , start_neurons * 1, name='deconv_1a')
    flops+= flop
    uconv1a_3 = concatenate([uconv1a_3, conv1_5, conv1a_5])
    uconv1_3, uconv1_5, flop = conv_op(uconv1a_3 , start_neurons * 1, name='deconv_1b')
    flops+= flop
    flops+= 3*3*uconv1_3.shape[1]*uconv1_3.shape[2]*start_neurons * 1
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1_3)  # D1

    # Feature-to-mask pipeline
    flops+= 3*3*uconv4_5.shape[1]*uconv4_5.shape[2]*start_neurons * 8
    uconv4_5 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same",activation=act1)(concatenate([uconv4_5, uconv4a_5]))
    flops+= 3*3*uconv3_5.shape[1]*uconv3_5.shape[2]*start_neurons * 4
    uconv3_5 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same",activation=act1)(concatenate([uconv4_5, uconv3_5, uconv3a_5]))
    flops+= 3*3*uconv2_5.shape[1]*uconv2_5.shape[2]*start_neurons * 2
    uconv2_5 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same",activation=act1)(concatenate([uconv3_5, uconv2_5, uconv2a_5]))
    flops+= 3*3*uconv1_5.shape[1]*uconv1_5.shape[2]*start_neurons * 1
    uconv1_5 = Conv2D(start_neurons * 1, (3, 3), padding="same",activation=act1)(concatenate([uconv2_5, uconv1_5, uconv1a_5]))
    flops+= 3*3*uconv1_5.shape[1]*uconv1_5.shape[2]*start_neurons * 1
    uconv1_5 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1_5)

    uconv1 = add([uconv1, uconv1_5])
    
    flops+= 1*1*uconv1.shape[1]*uconv1.shape[2]*1
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid", name= 'out')(uconv1)  # M

    model = Model(input_layer, output_layer)

       
    print('Total flops --->', flops)
    return model

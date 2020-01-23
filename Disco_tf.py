import tensorflow as tf

from keras.losses import categorical_crossentropy

# custom loss for mass decorrelation
def mass_decorrelation_loss(y_true, y_pred, sample_weight=None):

    # values of ~0.5-0.7 for 3 cat
    # I'm using an old keras, really need to go to tensorflow 2.0
    #cce = categorical_crossentropy(y_true[:,:-1], y_pred[:,:-1], sample_weight=sample_weight)
    cce = categorical_crossentropy(y_true[:,:-1], y_pred[:,:-1])
    lamb = 1e3 # TODO: tune
    # here first argument is the mass, second argument is output to decorrelate (3rd element is TauHTauH), third is weight
    if sample_weight is None: sample_weight = tf.ones_like(y_true[:,-1])
    disco = distance_corr(y_true[:,-1], y_pred[:,2], sample_weight)

    return cce + lamb * disco

def distance_corr(var_1, var_2, normedweight, power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])
 
    yy = tf.transpose(xx)
    amat = tf.math.abs(xx-yy)
    
    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
    
    yy = tf.transpose(xx)
    bmat = tf.math.abs(xx-yy)
   
    amatavg = tf.reduce_mean(amat*normedweight, axis=1)
    bmatavg = tf.reduce_mean(bmat*normedweight, axis=1)
 
    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg*normedweight)

    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg*normedweight)

    ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)
   
    if power==1:
        dCorr = tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    elif power==2:
        dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
    else:
        dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)))**power
  
    return dCorr
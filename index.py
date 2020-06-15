import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt



# def plot_image(image):
#     plt.imshow(image.reshape(28,28),cmap='binary')
#     plt.show()

#模型所需函数
def model(x,w,b):
    pred = tf.matmul(x,w)+b
    return tf.nn.softmax(pred)


def loss(x,y,w,b):
    pred = model(x,w,b)
    loss_ = tf.keras.losses.categorical_crossentropy(y_true=y,y_pred=pred)
    return tf.reduce_mean(loss_)
def grad(x,y,w,b):
    with tf.GradientTape() as tape:
        loss_ = loss(x,y,w,b)
    return tape.gradient(loss_,[w,b])

def accuracy(x,y,w,b):
    pred = model(x,w,b)
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#导入训练集和测试集
train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')
# print(train.head())
# print(test.head())

#导出训练集标签值
train_labels = train['label'].values
train_labels = np.array(train_labels)
# print(train_labels.shape)

#导出测试集图像值
train.drop('label', axis = 1, inplace = True)
train_images = train.values
train_images = np.reshape(train_images,[60000,28,28])
# print(train_images.shape)


#导出训练集标签值
test_labels = test['label'].values
test_labels = np.array(test_labels)
# print(test_labels.shape)


#导出训练集图像值
test.drop('label', axis = 1, inplace = True)
test_images = test.values
test_images = np.reshape(test_images,[10000,28,28])
# print(test_images.shape)



# plot_image(train_images[0])

#划分训练测试数据集
total_num = len(train_images)
print(total_num)
valid_split = 0.2
train_num = int(total_num*(1-valid_split))

train_x = train_images[:train_num]
train_y = train_labels[:train_num]

valid_x = train_images[train_num:]
valid_y = train_labels[train_num:]

test_x = test_images
test_y = test_labels


train_x = train_x.reshape(-1,784)
valid_x = valid_x.reshape(-1,784)
test_x = test_x.reshape(-1,784)


#数据归一化
train_x = tf.cast(train_x/255.0,tf.float32)
valid_x = tf.cast(valid_x/255.0,tf.float32)
test_x = tf.cast(test_x/255.0,tf.float32)

#独热编码
train_y = tf.one_hot(train_y,depth=10)
valid_y = tf.one_hot(valid_y,depth=10)
test_y = tf.one_hot(test_y,depth=10)




#建立模型
W = tf.Variable(tf.random.normal([784,10],mean=0.0,stddev=1.0,dtype=tf.float32))
B = tf.Variable(tf.zeros([10]),dtype=tf.float32)   #偏值项

training_epochs = 30
batch_size = 50
learning_rate = 0.005

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)   #优化器

total_step = int(train_num/batch_size)

loss_list_train = []
loss_list_valid = []
acc_list_train = []
acc_list_valid = []

for epoch in range(training_epochs):
    for step in range(total_step):
        xs = train_x[step*batch_size:(step+1)*batch_size]
        ys = train_y[step*batch_size:(step+1)*batch_size]

        grads = grad(xs,ys,W,B)
        optimizer.apply_gradients(zip(grads,[W,B]))
    loss_train = loss(train_x,train_y,W,B).numpy()
    loss_valid = loss(valid_x,valid_y,W,B).numpy()
    acc_train = accuracy(train_x,train_y,W,B).numpy()
    acc_valid = accuracy(valid_x,valid_y,W,B).numpy()
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    acc_list_train.append(acc_train)
    acc_list_valid.append(acc_valid)

    print("epoch={:3d},train_loss={:.4f},train_acc={:.4f},val_loss={:.4f},val_acc{:.4f}".format(epoch+1,loss_train,acc_train,loss_valid,acc_valid))


 #训练结果   
acc_test = accuracy(test_x,test_y,W,B).numpy()
print('learning rate',learning_rate)
print("Test Accuracy:",acc_test)




#预测模型
def predict(x,w,b):
    pred = model(x,w,b)
    result = tf.argmax(pred,1).numpy()
    return result


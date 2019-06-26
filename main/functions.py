import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

#从文件中获取columns的数列
def init_column(rawpath):
    columns_read = open(rawpath, 'r')
    columns_line = columns_read.readline()
    # 截掉末尾的换行符
    if columns_line[-1] == '\n':
        columns_line = columns_line[:-1]
    # columns数列
    columns_name = columns_line.split('	')
    return columns_name

# 过滤sql查询出来的数据 包含null的数据不使用，字段大于100000的数据不使用
def filter_data(rawpath, datapath, columns_name):
    raw_data_read = open(rawpath, 'r')
    data_write = open(datapath, 'w')

    raw_data_lines = raw_data_read.readlines()
    # 移除column 名字
    raw_data_lines.remove(raw_data_lines[0])
    for raw_data_line in raw_data_lines:
        # 截掉末尾的换行符
        if raw_data_line[-1] == '\n':
            raw_data_line = raw_data_line[:-1]

        # 一行原数据转换为数据列表
        raw_data_arr = raw_data_line.split('	')
        data_arr = []

        # 筛选元数据，字段大于100000和包含null，直接删掉不使用
        check_result = True

        if 'NULL' in raw_data_arr:
            check_result = False
        else:
            for raw_data in raw_data_arr:
                if float(raw_data) > 100000:
                    check_result = False
                    break

        # 将元数据转换为float类型，便于计算
        if check_result:
            for index in range(len(raw_data_arr)):
                data_arr.append(str(float(raw_data_arr[index])))

        if len(data_arr) == len(columns_name):
            data_write.write('  '.join(data_arr))
            data_write.write('\n')

    raw_data_read.close()
    data_write.close()
    return datapath

# 使用pandas可帮助我们生成这样的数据
#         blockcnt  blocktime  b.status  ...  pingtime  b.dnstime  tlstime
# 0            0.0        0.0       1.0  ...       0.0        4.0    316.0
# 1            0.0        0.0       1.0  ...       0.0        0.0      0.0
# 2            0.0        0.0       1.0  ...       0.0        7.0    240.0
# 3            0.0        0.0       1.0  ...       0.0        0.0      0.0
# 4            0.0        0.0       1.0  ...       0.0        0.0      0.0
def pd_read_csv(path, columns_name):
    pd_dataset = pd.read_csv(path, names=columns_name,
                             na_values="?", comment='\t',
                             sep=" ", skipinitialspace=True)
    pd_dataset.isna().sum()
    pd_dataset.dropna()
    return pd_dataset.copy()

# 创建model
def build_model(train_dataset):
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    print(epoch % 100)

# 训练model
def fit_model(model, train_dataset, train_labels):
    EPOCHS = 5
    history = model.fit(
        train_dataset, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])
    return history

def predict(model, dataset, labels):
    test_predictions = model.predict(dataset).flatten()
    plt.scatter(labels, test_predictions)
    plt.xlabel('True Values [blockcnt]')
    plt.ylabel('Predictions [blockcnt]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])

    # 查看错误
    error = test_predictions - labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [blockcnt]")
    _ = plt.ylabel("Count")

    plt.show()


# 画出数据
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [blockcdn]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$blockcdn^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()
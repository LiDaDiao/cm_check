import pathlib
import seaborn as sns

# https://github.com/LiDaDiao/cm_check.git

import functions as fun

def norm(x,train_stats):
  return (x - train_stats['mean']) / train_stats['std']

def run():
    # 原始数据路径
    raw_path_140004_130001 = 'raw_140004_130001.txt'
    # 过滤后生成数据路径
    data_path_140004_130001 = 'data_140004_130001.txt'
    # column数列
    # ['blockcnt', 'blocktime', 'b.status', 'alltime', 'contime', 'pinggoogle', 'pingtime', 'b.dnstime', 'tlstime']
    columns_name_140004_130001 = fun.init_column(raw_path_140004_130001)
    print(columns_name_140004_130001)

    # 过滤后生成数据路径
    data_path_140004_130001 = fun.filter_data(raw_path_140004_130001, data_path_140004_130001, columns_name_140004_130001)
    data_set_140004_130001 = fun.pd_read_csv(data_path_140004_130001, columns_name_140004_130001)
    # print(data_set_140004_130001)

    #将预测的标签与其他标签分开
    # data_set_140004_130001.pop('blockcnt')
    # data_set_140004_130001.pop('blocktime')

    #10%数据作为测试数据，其他数据为训练数据
    train_dataset_140004_130001 = data_set_140004_130001.sample(frac=0.9, random_state=0)
    # print(train_dataset_140004_130001)
    test_dataset_140004_130001 = data_set_140004_130001.drop(train_dataset_140004_130001.index)

    train_dataset_140004_130001.pop('blocktime')
    test_dataset_140004_130001.pop('blocktime')

    train_labels_140004_130001 = train_dataset_140004_130001.pop('blockcnt')
    test_labels_140004_130001 = test_dataset_140004_130001.pop('blockcnt')

    #检查一下训练数据，diag_kind="kde"是指画图类型
    # print(train_dataset_140004_130001[['blocktime', 'alltime', 'b.dnstime', 'tlstime']])
    # sns.pairplot(train_dataset_140004_130001[['blocktime', 'alltime', 'b.dnstime', 'tlstime']])
    # plt.show()

    # 生成描述性统计数据，总结数据集分布的集中趋势，分散和形状，不包括NaN值。
    # train_stats_140004_130001 = train_dataset_140004_130001.describe()
    # print(train_stats_140004_130001)
    # 行 列 调换
    # train_stats_140004_130001 = train_stats_140004_130001.transpose()
    # print(train_stats_140004_130001)

    # normed_train_data_140004_130001 = norm(train_dataset_140004_130001,train_stats_140004_130001)
    # normed_test_data_140004_130001 = norm(test_dataset_140004_130001)

    # 创建model，
    model = fun.build_model(train_dataset_140004_130001)
    # 检查model
    # model.summary()

    # 试下这个模型咋样
    # example_batch = train_dataset_140004_130001[:10]
    # print(example_batch)
    # example_result = model.predict(example_batch)
    # print(example_result)

    # 训练模型
    history = fun.fit_model(model, train_dataset_140004_130001, train_labels_140004_130001)

    # 画出训练模型
    # fun.plot_history(history)
    # hist = pd.DataFrame(history.history)
    # hist['epoch'] = history.epoch
    # hist.tail()
    # print(hist)
    loss, mae, mse = model.evaluate(test_dataset_140004_130001, test_labels_140004_130001, verbose=0)
    # print("Testing set loss {} blokcnt".format(loss))
    # print("Testing set mae {} blokcnt".format(mae))
    # print("Testing set mse {} blokcnt".format(mse))
    # print("Testing set Mean Abs Error: {:5.2f} blokcnt".format(mae))

    # 预测
    fun.predict(model, test_dataset_140004_130001, test_labels_140004_130001)

if __name__ == '__main__':
	run()

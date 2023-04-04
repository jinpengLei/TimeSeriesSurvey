import argparse
import os
import sys
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from op.mpa import Mpa, QIMpa
from log.Logger import Logger
def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--model', type=str, default='FEDformer',
                        help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


    parser.add_argument('--input_size', type=int, default=7, help='lstm input size')
    parser.add_argument('--output_size', type=int, default=7, help='lstm output size')
    parser.add_argument('--binary', type=int, default=1, help='normal lstm or bi-lstm')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size of lstm')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str, default='tanh'

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

    parser.add_argument('--mpa', type=int, default=0, help='use mpa select hyperparameters')

    parser.add_argument('--hidden_state_features', type=int, default=12,
                        help='number of features in LSTMs hidden states')
    parser.add_argument('--num_layers_lstm', type=int, default=1,
                        help='num of lstm layers')
    parser.add_argument('--hidden_state_features_uni_lstm', type=int, default=1,
                        help='number of features in LSTMs hidden states for univariate time series')
    parser.add_argument('--num_layers_uni_lstm', type=int, default=1,
                        help='num of lstm layers for univariate time series')
    parser.add_argument('--attention_size_uni_lstm', type=int, default=10,
                        help='attention size for univariate lstm')
    parser.add_argument('--hidCNN', type=int, default=10,
                        help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=100,
                        help='number of RNN hidden units')
    parser.add_argument('--window', type=int, default=24 * 7,
                        help='window size')
    parser.add_argument('--CNN_kernel', type=int, default=1,
                        help='the kernel size of the CNN layers')
    parser.add_argument('--highway_window', type=int, default=24,
                        help='The window size of the highway component')
    parser.add_argument('--hidSkip', type=int, default=5)
    parser.add_argument('--skip', type=float, default=24)
    parser.add_argument('--output_fun', type=str, default='sigmoid')

    args = parser.parse_args()

    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
        'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
        'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
        'TK': {'data': 'train_env.csv', 'T': 'xfwd', 'M': [4, 4, 4], 'S': [1, 1, 1], 'MS': [4, 4, 1]},
        'ELE': {'data': 'electricity.txt', 'T': 320, 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
        'SOLAR': {'data': 'solar_AL.txt', 'T': 136, 'M':[137, 137, 137], 'S': [1, 1, 1], 'MS':[137, 137, 1]},
        'EXC': {'data': 'exchange_rate.txt', 'T': 7, 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1]},
        'TRAFFIC': {'data': 'traffic.txt', 'T': 861, 'M': [862, 862, 862], 'S': [1, 1, 1], 'MS': [862, 862, 1]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        if args.model == 'Lstm' or args.model == 'TpaLstm':
            args.input_size, _, args.output_size = data_info[args.features]
        else:
            args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    lg = Logger('log/exp_record.log', level='debug')
    lg.logger.info("==============start Exp====================")
    is_use_mpa_str = ""
    if args.mpa == 0:
        is_use_mpa_str = "withoutMPA"
    elif args.mpa == 1:
        is_use_mpa_str = "useMPA"
    elif args.mpa == 2:
        is_use_mpa_str = "useQIMPA"
    else:
        is_use_mpa_str = "other"
    lg.logger.info("============{}_{}_{}====================".format(args.model, is_use_mpa_str, args.data))
    Exp = Exp_Main
    if args.model != "Lstm" and args.model != 'TpaLstm':
        if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.task_id,
                    args.model,
                    args.mode_select,
                    args.modes,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    ii)

                exp = Exp(args)  # set experiments
                exp.prepare_data()
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

                torch.cuda.empty_cache()
        else:
            ii = 0
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                          args.model,
                                                                                                          args.data,
                                                                                                          args.features,
                                                                                                          args.seq_len,
                                                                                                          args.label_len,
                                                                                                          args.pred_len,
                                                                                                          args.d_model,
                                                                                                          args.n_heads,
                                                                                                          args.e_layers,
                                                                                                          args.d_layers,
                                                                                                          args.d_ff,
                                                                                                          args.factor,
                                                                                                          args.embed,
                                                                                                          args.distil,
                                                                                                          args.des, ii)

            exp = Exp(args)  # set experiments
            exp.prepare_data()
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()
    elif args.model == 'Lstm' or args.model == 'TpaLstm':
        if args.mpa == 0:
            print("without mpa")
            if args.is_training:
                setting = '{}_{}_{}_{}_{}_sl{}_ll{}_pl{}_is{}_hs{}_ops{}_bi{}_eb{}_dt{}_{}'.format(
                    args.task_id,
                    args.model,
                    is_use_mpa_str,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.input_size,
                    args.hidden_size,
                    args.output_size,
                    args.binary,
                    args.embed,
                    args.distil,
                    args.des)
                exp = Exp(args)
                exp.prepare_data()
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

                torch.cuda.empty_cache()
            else:
                setting = '{}_{}_{}_{}_{}_sl{}_ll{}_pl{}_is{}_hs{}_ops{}_bi{}_eb{}_dt{}_{}'.format(
                    args.task_id,
                    args.model,
                    is_use_mpa_str,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.input_size,
                    args.hidden_size,
                    args.output_size,
                    args.binary,
                    args.embed,
                    args.distil,
                    args.des)

                exp = Exp(args)
                exp.prepare_data()
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, test=1)
                torch.cuda.empty_cache()
        else:
            setting = '{}_{}_{}_{}_is{}_ops{}_bi{}_eb{}_dt{}_{}'.format(
                args.model,
                is_use_mpa_str,
                args.data,
                args.features,
                args.input_size,
                args.output_size,
                args.binary,
                args.embed,
                args.distil,
                args.des
            )
            exp = Exp(args)
            exp.prepare_data()
            func = lambda hyperparameters_list: fitFunc(exp, setting, hyperparameters_list)
            search_agents_no = 5
            max_iter = 5
            if args.model == 'Lstm':
                dim = 2
                ub = [300, 0.05]
                lb = [100, 0.001]
            else:
                dim = 5
                ub = [50, 0.05, 50, 50, 300]
                lb = [10, 0.001, 8, 8, 60]
            if args.mpa == 1:
                mpa = Mpa(search_agents_no=1, max_iter=1, dim=dim, ub=ub, lb=lb, fobj=func)
            elif args.mpa == 2:
                mpa = QIMpa(search_agents_no=10, max_iter=10, dim=dim, ub=ub, lb=lb, fobj=func)
            else:
                pass
            [best_score, best_pos, convergence_curve] = mpa.opt()
            print(best_pos)
            print(best_score)
            print(convergence_curve)
            path = os.path.join("best_result", setting)
            file_path = path + "/" + "best_record.npy"
            min_mse, min_mae = np.load(file_path)
            print(min_mse)
            print(min_mae)
            log = Logger('log/best_result', level='debug')
            log.logger.info("============{}==============".format(setting))
            log.logger.info("==========best pos===========")
            log.logger.info(best_pos)
            log.logger.info("==========best score===========")
            log.logger.info(best_score)
            log.logger.info(convergence_curve)
            log.logger.info("===========min mse mae===============")
            log.logger.info("{}  {}".format(min_mae, min_mae))
    else:
        if args.mpa == 0:
            print("without mpa")
            if args.is_training:
                setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_is{}_hs{}_ops{}_bi{}_eb{}_dt{}_{}'.format(
                    args.task_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.input_size,
                    args.hidden_state_features,
                    args.output_size,
                    args.binary,
                    args.embed,
                    args.distil,
                    args.des)
                exp = Exp(args)
                exp.prepare_data()
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

                torch.cuda.empty_cache()
            else:
                setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_is{}_hs{}_ops{}_bi{}_eb{}_dt{}_{}'.format(
                    args.task_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.input_size,
                    args.hidden_size,
                    args.output_size,
                    args.binary,
                    args.embed,
                    args.distil,
                    args.des)

                exp = Exp(args)
                exp.prepare_data()
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, test=1)
                torch.cuda.empty_cache()

    lg.logger.info("==============end Exp====================")

def update_hyparameter(args, hyperparameters_list):
    if args.model == 'Lstm':
        args.hidden_size = round(hyperparameters_list[0])
        args.learning_rate = hyperparameters_list[1]

        print("hidden_size")
        print(args.hidden_size)
        print("learning_rate")
        print(args.learning_rate)
    else:
        args.hidden_state_feature = round(hyperparameters_list[0])
        args.learning_rate = hyperparameters_list[1]
        args.attention_size_uni_lstm = round(hyperparameters_list[2])
        args.hidCNN = round(hyperparameters_list[3])
        args.hidRNN = round(hyperparameters_list[4])

        print("hidden state feature size")
        print(args.hidden_state_feature)
        print("learning rate")
        print(args.learning_rate)
        print("attention size")
        print(args.attention_size_uni_lstm)
        print("hidCNN")
        print(args.hidCNN)
        print("hidRNN")
        print(args.hidRNN)
    return args


def fitFunc(exp, setting, hyperparameters_list):
    args = update_hyparameter(exp.args, hyperparameters_list)
    exp.update_args(args=args)
    exp._build_model()
    exp.train(setting)
    now_test_mse, now_test_mae = exp.test(setting)
    path = os.path.join("best_result", setting)
    file_path = path + "/" + "best_record.npy"
    if not os.path.exists(path):
        os.makedirs(path)
        np.save(file_path, [now_test_mse, now_test_mae])
    else:
        now_best_mse, now_best_mae = np.load(file_path)
        if(now_test_mse < now_best_mse):
            np.save(file_path, [now_test_mse, now_test_mae])
    return now_test_mse


if __name__ == "__main__":
    main()

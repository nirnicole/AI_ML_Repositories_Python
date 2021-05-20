"""
Course: Biological computation
Name: Nir nicole
Module: main execution script
"""
import GOLutils
import NNframe
import xlsxwriter

def run_test(l_hidden=1, n_hidden=4, l_rate=0.5, b_ratio=4, e_count=200, test_allow=False, binar_act=False):
    """
    test procedure.

    # network variables - tunable, i gave here another interface to tune them according to notes.
    :param l_hidden: multiple layers give more parameters(curves) to the function.
    :param n_hidden: same, gives more parameters to the function. remeber there is another implicit one, a bias.
    :param l_rate:   step size, too low - small steps and may not converge, too high - big steps and may miss the local minimum.
    :param b_ratio:  split the set into x batches.
    :param e_count:  epochs count to limit the training per epochs amount. can(and should) be limited by progress instead!
    :param test_allow:
    :return:
    """
    # fetching data and translating it.
    training_size = 80      # training set, ff and bp on the net while adjusting weigths by its predictions.
    validation_size = 40    # training validation set. the net will never 'know' this set and it wont effect it weights(only for progress tracing and overfitting alert)
    test_size = 24          # the rest of the data base, we can test the net on this set as well to see more "real world" results.
    oscilators_data = GOLutils.get_dataset("dataset_oscilators.txt")
    nonoscilators_data = GOLutils.get_dataset("dataset_non.txt")
    train_data, train_predictions, validation_data, validation_predictions, test_data, test_predictions = GOLutils.split_sets(oscilators_data, nonoscilators_data,training_size,validation_size,test_size)

    # network variables - DO NOT TUNE!
    n_inputs = len(train_data[0])         #tuned automatically.
    n_outputs = 2                         #the neuron that will shoot indicats the classification.

    # create the network - you can tune some of its parameters from outside.
    network = NNframe.NeuralNetwork(n_inputs, n_hidden, n_outputs, l_hidden, l_rate, e_count,True,binar_act)

    # training the net while providing loss tracing stats for farther review.
    print("\nTraining net:")
    epoch_errors = network.train(train_data, train_predictions,validation_data, validation_predictions, b_ratio)
    #network.print_net()

    # testing the net on the whole data base
    if test_allow:
        print("\nFeed forward training_set on the net:")
        results_trainset = network.test_net(train_data, train_predictions, False)
        print("\nFeed forward validation_set on the net:")
        results_valset = network.test_net(validation_data, validation_predictions, False)
        print("\nFeed forward test_set on the net:")
        results_test = network.test_net(test_data, test_predictions, False)
        return results_trainset, results_valset, results_test

    return epoch_errors

if __name__ == "__main__":
    """
    tests and statistics controlled from here.    
    """

    #tests run-flags
    costum_tests = True                 #you can switch my tests on from here
    binar_activation_tests = False        #you can switch my tests on from here

    if binar_activation_tests:
        # export eror statistics
        with xlsxwriter.Workbook('stats_binarActivation.xlsx') as workbook:

            ###############################################################################################
            # RESULTS TRACES:
            # track after results (no loss traces, True flag makes run_test() return results here)
            worksheet0 = workbook.add_worksheet('results')
            worksheet0.write_row(0 ,0 ,("attampt:","train_set","","","validation_set","","","test_set"))
            worksheet0.write_row(1 ,1 ,("got:", "out of:", "certainty:")*3)
            worksheet0.write_column(2 ,0 ,("1-layer",))

            results_trainset, results_valset, results_test = run_test(1, 4, 0.5, 4, 100,True,True)
            worksheet0.write_row(2 ,1 ,results_trainset)
            worksheet0.write_row(2 ,4 ,results_valset)
            worksheet0.write_row(2 ,7 ,results_test)
            ###############################################################################################
            # LOSS TRACES:
            # Overfitting check stats
            worksheet1 = workbook.add_worksheet('loss_stats')
            worksheet1.write_row(0 ,0 ,("Epoch:",))
            worksheet1.write_row(1 ,1 ,("train_loss", "validation_loss"))
            worksheet1.write_column(1 ,0 ,list(range(201)))
            for index in range(1):
                train_loss, val_loss = run_test(1, 4, 0.5, 4, 200,False,True)
                worksheet1.write_column(2 , (index+1) ,train_loss)
                worksheet1.write_column(2 , (index+2) ,val_loss)

    if costum_tests:
        # export eror statistics
        with xlsxwriter.Workbook('stats.xlsx') as workbook:

            ###############################################################################################
            # RESULTS TRACES:
            # track after results (no loss traces, True flag makes run_test() return results here)
            worksheet0 = workbook.add_worksheet('diffrent_strcutures')
            worksheet0.write_row(0 ,0 ,("attampt:","train_set","","","validation_set","","","test_set"))
            worksheet0.write_row(1 ,1 ,("got:", "out of:", "certainty:")*3)
            worksheet0.write_column(2 ,0 ,("1-layer, 4-neurons", "2-layer, 8-neurons", "3-layer, 6-neurons", "1-layer, 20-neurons", "1/4, no batchs", "1/4, 0.1 l_rate", "1/4, 0.9 l_rate"))

            results_trainset, results_valset, results_test = run_test(1, 4, 0.5, 4, 100,True)
            worksheet0.write_row(2 ,1 ,results_trainset)
            worksheet0.write_row(2 ,4 ,results_valset)
            worksheet0.write_row(2 ,7 ,results_test)
            results_trainset, results_valset, results_test = run_test(2, 8, 0.5, 4, 200,True)
            worksheet0.write_row(3 ,1 ,results_trainset)
            worksheet0.write_row(3 ,4 ,results_valset)
            worksheet0.write_row(3 ,7 ,results_test)
            results_trainset, results_valset, results_test = run_test(3, 6, 0.5, 4, 350,True)
            worksheet0.write_row(4 ,1 ,results_trainset)
            worksheet0.write_row(4 ,4 ,results_valset)
            worksheet0.write_row(4 ,7 ,results_test)
            results_trainset, results_valset, results_test = run_test(1, 20, 0.5, 4, 200,True)
            worksheet0.write_row(5 ,1 ,results_trainset)
            worksheet0.write_row(5 ,4 ,results_valset)
            worksheet0.write_row(5 ,7 ,results_test)
            results_trainset, results_valset, results_test = run_test(1, 4, 0.5, 1, 100,True)
            worksheet0.write_row(6 ,1 ,results_trainset)
            worksheet0.write_row(6 ,4 ,results_valset)
            worksheet0.write_row(6 ,7 ,results_test)
            results_trainset, results_valset, results_test = run_test(1, 4, 0.1, 4, 100,True)
            worksheet0.write_row(7 ,1 ,results_trainset)
            worksheet0.write_row(7 ,4 ,results_valset)
            worksheet0.write_row(7 ,7 ,results_test)
            results_trainset, results_valset, results_test = run_test(1, 4, 0.9, 4, 100,True)
            worksheet0.write_row(8 ,1 ,results_trainset)
            worksheet0.write_row(8 ,4 ,results_valset)
            worksheet0.write_row(8 ,7 ,results_test)

            ###############################################################################################
            # LOSS TRACES:
            # Overfitting check stats
            worksheet1 = workbook.add_worksheet('loss_stats')
            worksheet1.write_row(0 ,0 ,("Epoch:",))
            worksheet1.write_row(1 ,1 ,("train_loss", "validation_loss"))
            worksheet1.write_column(1 ,0 ,list(range(201)))
            for index in range(1):
                train_loss, val_loss = run_test(1, 4, 0.5, 4, 200)
                worksheet1.write_column(2 , (index+1) ,train_loss)
                worksheet1.write_column(2 , (index+2) ,val_loss)

            # hidden layers neuron size
            worksheet2 = workbook.add_worksheet('hidden_size')
            worksheet2.write_row(0 ,0 ,("Epoch:","layer size:",4,"layer size:",8,"layer size:",12,"layer size:",16))
            worksheet2.write_row(1 ,1 ,("train_loss", "validation_loss")*4)
            worksheet2.write_column(1 ,0 ,list(range(201)))
            for index in range(4):
                train_loss, val_loss = run_test(1, (index + 1) * 4, 0.5, 4, 200)
                worksheet2.write_column(2 , (2*index+1) ,train_loss)
                worksheet2.write_column(2 , (2*index+2) ,val_loss)

            # hidden layers depth
            worksheet3 = workbook.add_worksheet('hidden_depth')
            worksheet3.write_row(0 ,0 ,("Epoch:","depth:",1,"depth:",2,"depth:",3,"depth:",4))
            worksheet3.write_row(1 ,1 ,("train_loss", "validation_loss")*4)
            worksheet3.write_column(1 ,0 ,list(range(201)))
            for index in range(4):
                train_loss, val_loss = run_test((index + 1), 4, 0.5, 4, 200)
                worksheet3.write_column(2 , (2*index+1) ,train_loss)
                worksheet3.write_column(2 , (2*index+2) ,val_loss)

            # adjusting the leraning rate
            worksheet4 = workbook.add_worksheet('learning_rate')
            worksheet4.write_row(0 ,0 ,("Epoch:","learning_rate:",0.1,"learning_rate:",0.2,"learning_rate:",0.3,"learning_rate:",0.4,"learning_rate:",0.5,"learning_rate:",0.6,"learning_rate:",0.7,"learning_rate:",0.8,"learning_rate:",0.9))
            worksheet4.write_row(1 ,1 ,("train_loss", "validation_loss")*9)
            worksheet4.write_column(1 ,0 ,list(range(201)))
            for index in range(9):
                train_loss, val_loss = run_test(1, 4 , float((index + 1)/10), 4, 200)
                worksheet4.write_column(2 , (2*index+1) ,train_loss)
                worksheet4.write_column(2 , (2*index+2) ,val_loss)


            # diffrent batch sizes
            worksheet5 = workbook.add_worksheet('batch_size')
            worksheet5.write_row(0 ,0 ,("Epoch:","batch_ratio:",1,"batch_ratio:",2,"batch_ratio:",4))
            worksheet5.write_row(1 ,1 ,("train_loss", "validation_loss")*3)
            worksheet5.write_column(1 ,0 ,list(range(201)))
            for index in range(3):
                train_loss, val_loss = run_test(1, 4, 0.5, max(1,2*index), 200)
                worksheet5.write_column(2 , (2*index+1) ,train_loss)
                worksheet5.write_column(2 , (2*index+2) ,val_loss)


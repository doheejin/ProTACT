import os
import time
import argparse
import random
import numpy as np
from models.ProTACT import build_ProTACT
import tensorflow as tf
from configs.configs import Configs
from utils.read_data_pr import read_pos_vocab, read_word_vocab, read_prompts_we, read_essays_prompts, read_prompts_pos
from utils.general_utils import get_scaled_down_scores, pad_hierarchical_text_sequences, get_attribute_masks, load_word_embedding_dict, build_embedd_table
from evaluators.multitask_evaluator_all_attributes import Evaluator as AllAttEvaluator
from tensorflow import keras
import matplotlib.pyplot as plt

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []        
        
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

def main():
    parser = argparse.ArgumentParser(description="ProTACT model")
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--model_name', type=str,
                        choices=['ProTACT'],
                        help='name of model')
    parser.add_argument('--num_heads', type=int, default=2, help='set the number of heads in Multihead Attention')
    parser.add_argument('--features_path', type=str, default='data/hand_crafted_v3.csv')
    args = parser.parse_args()
    test_prompt_id = args.test_prompt_id
    seed = args.seed
    num_heads = args.num_heads
    features_path = args.features_path + str(test_prompt_id) + '.csv'

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print("Test prompt id is {} of type {}".format(test_prompt_id, type(test_prompt_id)))
    print("Seed: {}".format(seed))

    configs = Configs()

    data_path = configs.DATA_PATH
    train_path = data_path + str(test_prompt_id) + '/train.pk'
    dev_path = data_path + str(test_prompt_id) + '/dev.pk'
    test_path = data_path + str(test_prompt_id) + '/test.pk'
    pretrained_embedding = configs.PRETRAINED_EMBEDDING
    embedding_path = configs.EMBEDDING_PATH
    readability_path = configs.READABILITY_PATH
    prompt_path = configs.PROMPT_PATH
    vocab_size = configs.VOCAB_SIZE
    epochs = configs.EPOCHS
    batch_size = configs.BATCH_SIZE
    print("Numhead : ", num_heads, " | Features : ", features_path, " | Pos_emb : ", configs.EMBEDDING_DIM)

    read_configs = {
        'train_path': train_path,
        'dev_path': dev_path,
        'test_path': test_path,
        'features_path': features_path,
        'readability_path': readability_path,
        'vocab_size': vocab_size
    }
    # read POS for prompts
    pos_vocab = read_pos_vocab(read_configs)
    prompt_pos_data = read_prompts_pos(prompt_path, pos_vocab) # for prompt POS embedding 

    # read words for prompts 
    word_vocab = read_word_vocab(read_configs)
    prompt_data = read_prompts_we(prompt_path, word_vocab) # for prompt word embedding 
    
    # read essays and prompts
    train_data, dev_data, test_data = read_essays_prompts(read_configs, prompt_data, prompt_pos_data, pos_vocab) 

    if pretrained_embedding:
        embedd_dict, embedd_dim, _ = load_word_embedding_dict(embedding_path)
        embedd_matrix = build_embedd_table(word_vocab, embedd_dict, embedd_dim, caseless=True)
        embed_table = [embedd_matrix]
    else:
        embed_table = None

    max_sentlen = max(train_data['max_sentlen'], dev_data['max_sentlen'], test_data['max_sentlen'])
    max_sentnum = max(train_data['max_sentnum'], dev_data['max_sentnum'], test_data['max_sentnum'])
    prompt_max_sentlen = prompt_data['max_sentlen']
    prompt_max_sentnum = prompt_data['max_sentnum']

    print('max sent length: {}'.format(max_sentlen))
    print('max sent num: {}'.format(max_sentnum))
    print('max prompt sent length: {}'.format(prompt_max_sentlen))
    print('max prompt sent num: {}'.format(prompt_max_sentnum))

    train_data['y_scaled'] = get_scaled_down_scores(train_data['data_y'], train_data['prompt_ids'])
    dev_data['y_scaled'] = get_scaled_down_scores(dev_data['data_y'], dev_data['prompt_ids'])
    test_data['y_scaled'] = get_scaled_down_scores(test_data['data_y'], test_data['prompt_ids'])

    X_train_pos = pad_hierarchical_text_sequences(train_data['pos_x'], max_sentnum, max_sentlen)
    X_dev_pos = pad_hierarchical_text_sequences(dev_data['pos_x'], max_sentnum, max_sentlen)
    X_test_pos = pad_hierarchical_text_sequences(test_data['pos_x'], max_sentnum, max_sentlen)

    X_train_pos = X_train_pos.reshape((X_train_pos.shape[0], X_train_pos.shape[1] * X_train_pos.shape[2]))
    X_dev_pos = X_dev_pos.reshape((X_dev_pos.shape[0], X_dev_pos.shape[1] * X_dev_pos.shape[2]))
    X_test_pos = X_test_pos.reshape((X_test_pos.shape[0], X_test_pos.shape[1] * X_test_pos.shape[2]))

    X_train_prompt = pad_hierarchical_text_sequences(train_data['prompt_words'], max_sentnum, max_sentlen)
    X_dev_prompt = pad_hierarchical_text_sequences(dev_data['prompt_words'], max_sentnum, max_sentlen)
    X_test_prompt = pad_hierarchical_text_sequences(test_data['prompt_words'], max_sentnum, max_sentlen)

    X_train_prompt = X_train_prompt.reshape((X_train_prompt.shape[0], X_train_prompt.shape[1] * X_train_prompt.shape[2]))
    X_dev_prompt = X_dev_prompt.reshape((X_dev_prompt.shape[0], X_dev_prompt.shape[1] * X_dev_prompt.shape[2]))
    X_test_prompt = X_test_prompt.reshape((X_test_prompt.shape[0], X_test_prompt.shape[1] * X_test_prompt.shape[2]))

    X_train_prompt_pos = pad_hierarchical_text_sequences(train_data['prompt_pos'], max_sentnum, max_sentlen)
    X_dev_prompt_pos = pad_hierarchical_text_sequences(dev_data['prompt_pos'], max_sentnum, max_sentlen)
    X_test_prompt_pos = pad_hierarchical_text_sequences(test_data['prompt_pos'], max_sentnum, max_sentlen)

    X_train_prompt_pos = X_train_prompt_pos.reshape((X_train_prompt_pos.shape[0], X_train_prompt_pos.shape[1] * X_train_prompt_pos.shape[2]))
    X_dev_prompt_pos = X_dev_prompt_pos.reshape((X_dev_prompt_pos.shape[0], X_dev_prompt_pos.shape[1] * X_dev_prompt_pos.shape[2]))
    X_test_prompt_pos = X_test_prompt_pos.reshape((X_test_prompt_pos.shape[0], X_test_prompt_pos.shape[1] * X_test_prompt_pos.shape[2]))

    X_train_linguistic_features = np.array(train_data['features_x'])
    X_dev_linguistic_features = np.array(dev_data['features_x'])
    X_test_linguistic_features = np.array(test_data['features_x'])

    X_train_readability = np.array(train_data['readability_x'])
    X_dev_readability = np.array(dev_data['readability_x'])
    X_test_readability = np.array(test_data['readability_x'])

    Y_train = np.array(train_data['y_scaled'])
    Y_dev = np.array(dev_data['y_scaled'])
    Y_test = np.array(test_data['y_scaled'])

    X_train_attribute_rel = get_attribute_masks(Y_train)
    X_dev_attribute_rel = get_attribute_masks(Y_dev)
    X_test_attribute_rel = get_attribute_masks(Y_test)

    print('================================')
    print('X_train_pos: ', X_train_pos.shape)
    print('X_train_prompt_words: ', X_train_prompt.shape)
    print('X_train_prompt_pos: ', X_train_prompt_pos.shape)
    print('X_train_readability: ', X_train_readability.shape)
    print('X_train_ling: ', X_train_linguistic_features.shape)
    print('X_train_attribute_rel: ', X_train_attribute_rel.shape)
    print('Y_train: ', Y_train.shape)

    print('================================')
    print('X_dev_pos: ', X_dev_pos.shape)
    print('X_dev_prompt_words: ', X_dev_prompt.shape)
    print('X_dev_prompt_pos: ', X_dev_prompt_pos.shape)
    print('X_dev_readability: ', X_dev_readability.shape)
    print('X_dev_ling: ', X_dev_linguistic_features.shape)
    print('X_dev_attribute_rel: ', X_dev_attribute_rel.shape)
    print('Y_dev: ', Y_dev.shape)

    print('================================')
    print('X_test_pos: ', X_test_pos.shape)
    print('X_test_prompt_words: ', X_test_prompt.shape)
    print('X_test_prompt_pos: ', X_test_prompt_pos.shape)
    print('X_test_readability: ', X_test_readability.shape)
    print('X_test_ling: ', X_test_linguistic_features.shape)
    print('X_test_attribute_rel: ', X_test_attribute_rel.shape)
    print('Y_test: ', Y_test.shape)
    print('================================')

    train_features_list = [X_train_pos, X_train_prompt, X_train_prompt_pos, X_train_linguistic_features, X_train_readability]
    dev_features_list = [X_dev_pos, X_dev_prompt, X_dev_prompt_pos, X_dev_linguistic_features, X_dev_readability]
    test_features_list = [X_test_pos, X_test_prompt, X_test_prompt_pos, X_test_linguistic_features, X_test_readability]

    model = build_ProTACT(len(pos_vocab), len(word_vocab), max_sentnum, max_sentlen, 
                      X_train_readability.shape[1],
                      X_train_linguistic_features.shape[1],
                      configs, Y_train.shape[1], num_heads, embed_table)

    evaluator = AllAttEvaluator(test_prompt_id, dev_data['prompt_ids'], test_data['prompt_ids'], dev_features_list,
                                test_features_list, Y_dev, Y_test, seed)

    evaluator.evaluate(model, -1, print_info=True)

    custom_hist = CustomHistory() 
    custom_hist.init() 
    
    for ii in range(epochs):
        print('Epoch %s/%s' % (str(ii + 1), epochs))
        start_time = time.time()
        model.fit(
            train_features_list,
            Y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True, validation_data=(dev_features_list,Y_dev),callbacks=[custom_hist])
        tt_time = time.time() - start_time
        print("Training one epoch in %.3f s" % tt_time)
        evaluator.evaluate(model, ii + 1)
        print("Train Loss: ", custom_hist.train_loss[-1], "|| Val Loss: ", custom_hist.val_loss[-1])

    evaluator.print_final_info()

    '''# show the loss as the graph
    fig, loss_graph = plt.subplots()
    loss_graph.plot(custom_hist.train_loss,'y',label='train loss')
    loss_graph.plot(custom_hist.val_loss,'r',label='val loss')
    loss_graph.set_xlabel('epoch')
    loss_graph.set_ylabel('loss')
    plt.savefig(str('images/protact/test_prompt_'+ str(test_prompt_id) + '_seed_' + str(seed) + '_loss.png'))'''


if __name__ == '__main__':
    main()

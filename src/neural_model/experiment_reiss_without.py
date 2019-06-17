import sys
import collections
import numpy
import numpy as np
import random
import math
import os
import gc
import keras 
import ast
import codecs

from keras.preprocessing import sequence
from sequence import pad_sequences

try:
    import ConfigParser as configparser
except:
    import configparser

from Human_needs_elmo_without import Human_needs
from evaluator_reiss_know_c2c import MLTEvaluator

def read_input_files(file_path, max_sentence_length=-1):
    """
    Reads input files in tab-separated format.
    Will split file_paths on comma, reading from multiple files.
    """
    sentences = []
    labels = []
    line_length = None
    label_distribution=[]
    knowledge_per = []
    context_s = []
    knowledge = []
    story_id_know=[]
    knowledge_final = []
    lst2=[]
    x=''
    length=[]
    max_length=0
    sub_path=[]
    id=[]
    w=[]
    weight=[]
    weight_per = []
      
    if config["human_needs"] == "maslow":
        reiss = ['physiological', 'love', 'spiritual growth', 'esteem', 'stability']
    elif config["human_needs"] == "reiss":
      reiss=['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']
      #reiss = ['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'contact', 'savings', 'idealism', 'rest']
            
            
    with codecs.open(file_path,encoding="utf-8",mode="r") as f: 
        for line in f:
            story_id = line.split('\t')[0]
            sent = line.split('\t')[1].strip()
            context = line.split('\t')[1].replace('|', " ")
            char = line.split('\t')[3]
            label = line.split('\t')[-1].strip()
            lab = ast.literal_eval(label)
            pos = [i for i, j in enumerate(lab) if j == 1]
            label_distribution.append(label)
            sentences.append(char+'#'+sent)
            context_s.append(context) 
            id.append(story_id)
    batch = list(zip(sentences, context_s, label_distribution))

    
    return list(zip(id, sentences, context_s, label_distribution))

def padding(input, maxlen):
    """ 
    Padding the input sequence.....
    """
    
    sentences,knowledge,context_s,label_distribution = zip(*input) 
    sentences = keras.preprocessing.sequence.pad_sequences(list(sentences), padding='post', truncating='post', maxlen=maxlen)
    knowledge = keras.preprocessing.sequence.pad_sequences(knowledge, padding='post', truncating='post', maxlen=maxlen)
    context_s = keras.preprocessing.sequence.pad_sequences(context_s, padding='post', truncating='post', maxlen=maxlen)

    return list(zip(sentences,label_distribution))

def parse_config(config_section, config_path):
    """
    Reads configuration from the file and returns a dictionary.
    Tries to guess the correct datatype for each of the config values.
    """
    config_parser = configparser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()
    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif value.isdigit():
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config


def is_float(value):
    """
    Check in value is of type float()
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def create_batches_of_sentence_ids(sentences, batch_equal_size, max_batch_size):
    """
    Groups together sentences into batches
    If max_batch_size is positive, this value determines the maximum number of sentences in each batch.
    If max_batch_size has a negative value, the function dynamically creates the batches such that each batch contains abs(max_batch_size) words.
    Returns a list of lists with sentences ids.
    """
    batches_of_sentence_ids = []
    if batch_equal_size == True:
        sentence_ids_by_length = collections.OrderedDict()
        sentence_length_sum = 0.0
        for i in range(len(sentences)):
            length = len(sentences[i])
            if length not in sentence_ids_by_length:
                sentence_ids_by_length[length] = []
            sentence_ids_by_length[length].append(i)

        for sentence_length in sentence_ids_by_length:
            if max_batch_size > 0:
                batch_size = max_batch_size
            else:
                batch_size = int((-1 * max_batch_size) / sentence_length)

            for i in range(0, len(sentence_ids_by_length[sentence_length]), batch_size):
                batches_of_sentence_ids.append(sentence_ids_by_length[sentence_length][i:i + batch_size])
    else:
        current_batch = []
        max_sentence_length = 0
        for i in range(len(sentences)):
            current_batch.append(i)
            if len(sentences[i]) > max_sentence_length:
                max_sentence_length = len(sentences[i])
            if (max_batch_size > 0 and len(current_batch) >= max_batch_size) \
              or (max_batch_size <= 0 and len(current_batch)*max_sentence_length >= (-1 * max_batch_size)):
                batches_of_sentence_ids.append(current_batch)
                current_batch = []
                max_sentence_length = 0
        if len(current_batch) > 0:
            batches_of_sentence_ids.append(current_batch)
    return batches_of_sentence_ids



def process_sentences(data_train, data, model, is_training, learningrate, config, name, epoch):
    """
    Process all the sentences with the labeler, return evaluation metrics.
    """
    evaluator = MLTEvaluator(config)
    batches_of_sentence_ids = create_batches_of_sentence_ids(data, config["batch_equal_size"], config["max_batch_size"])
    if is_training == True:
        random.shuffle(batches_of_sentence_ids)

    for sentence_ids_in_batch in batches_of_sentence_ids:
        batch = [data_train[i] for i in sentence_ids_in_batch]
        cost, sentence_scores, token_scores = model.process_batch(data_train, batch, is_training, learningrate)
        evaluator.append_data(cost, batch, sentence_scores, token_scores, name, epoch)

    while config["garbage_collection"] == True and gc.collect() > 0:
            pass

    results = evaluator.get_results(name)
    for key in results:
       print(key + ": " + str(results[key]))
    if name =="test": 
          f1  = codecs.open("./result_with_know_c2c/result_test_"+str(epoch)+".txt",encoding='utf-8',mode='w')
          f1.write(str(results))
          f1.close()   
    if name =="dev":
          f1 = codecs.open("./result_with_know_c2c/result_train_"+str(epoch)+".txt",encoding="utf-8",mode="w")
          f1.write(str(results))
          f1.close()
    return results


def run_experiment(config_path):
    config = parse_config("config", config_path)
    temp_model_path = config_path + ".model"
    if "random_seed" in config:
        random.seed(config["random_seed"])
        numpy.random.seed(config["random_seed"])

    for key, val in config.items():
        print(str(key) + ": " + str(val))

    data_train, data_dev, data_test = None, None, None
    if config["path_train"] != None and len(config["path_train"]) > 0:
        data_train = read_input_files(config["path_train"], config["max_train_sent_length"])
    if config["path_dev"] != None and len(config["path_dev"]) > 0:
        data_dev = read_input_files(config["path_dev"])
    if config["path_test"] != None and len(config["path_test"]) > 0:
        data_test = []
        for path_test in config["path_test"].strip().split(":"):
            data_test += read_input_files(path_test)
    print("Word_embedding_size",config["word_embedding_size"])
    print("Word_embedding_path",config["preload_vectors"])
    
    model = Human_needs(config)
    model.build_vocabs(data_train, data_dev, data_test, config["word_embedding_size"], config["preload_vectors"])
    model.construct_network()
    model.initialize_session()
    if config["preload_vectors"] != None:
        model.preload_word_embeddings(config["preload_vectors"])

    print("parameter_count: " + str(model.get_parameter_count()))
    print("parameter_count_without_word_embeddings: " + str(model.get_parameter_count_without_word_embeddings()))

    if data_train != None:
        model_selector = config["model_selector"].split(":")[0]
        model_selector_type = config["model_selector"].split(":")[1]
        best_selector_value = 0.0
        best_epoch = -1
        learningrate = config["learningrate"]
        
        for epoch in range(config["epochs"]):
            print("EPOCH: " + str(epoch))
            print("current_learningrate: " + str(learningrate))
            random.shuffle(data_train)
            id, sentences, context_s, label_distribution = zip(*data_train) 
            results_train = process_sentences(data_train, sentences, model, is_training=True, learningrate=learningrate, config=config, name="train", epoch=epoch)
            
            if data_test != None:
                id, sentences, context_s, label_distribution = zip(*data_test) 
                f1  = codecs.open("./result_with_know_c2c/result_"+str(epoch)+".txt",encoding='utf-8',mode='w')
                f1.close()
                results_test = process_sentences(data_test, sentences, model, is_training=False, learningrate=0.0, config=config, name="test", epoch=epoch)
            
            if data_dev != None:
                id, sentences, context_s, label_distribution= zip(*data_dev) 
                results_dev = process_sentences(data_dev, sentences, model, is_training=False, learningrate=0.0, config=config, name="dev", epoch=epoch)
                
                if math.isnan(results_dev["dev_cost_sum"]) or math.isinf(results_dev["dev_cost_sum"]):
                    raise ValueError("Cost is NaN or Inf. Exiting.")
                    
                if (epoch == 0 or (model_selector_type == "high" and results_dev[model_selector] > best_selector_value) 
                               or (model_selector_type == "low" and results_dev[model_selector] < best_selector_value)):
                    best_epoch = epoch
                    best_selector_value = results_dev[model_selector]
                    model.saver.save(model.session, temp_model_path, latest_filename=os.path.basename(temp_model_path)+".checkpoint")
                    
                print("best_epoch: " + str(best_epoch))
                if config["stop_if_no_improvement_for_epochs"] > 0 and (epoch - best_epoch) >= config["stop_if_no_improvement_for_epochs"]:
                    break
                if (epoch - best_epoch) > 5:
                    learningrate *= config["learningrate_decay"]
                
            while config["garbage_collection"] == True and gc.collect() > 0:
                pass
        
        if data_dev != None and best_epoch >= 0:
            # loading the best model so far
            model.saver.restore(model.session, temp_model_path)
            os.remove(temp_model_path+".checkpoint")
            os.remove(temp_model_path+".data-00000-of-00001")
            os.remove(temp_model_path+".index")
            os.remove(temp_model_path+".meta")
        
    if config["save"] is not None and len(config["save"]) > 0:
        model.save(config["save"])

    if config["path_test"] is not None:
        i = 0
        
        for path_test in config["path_test"].strip().split(":"):
            data_test = read_input_files(path_test)
            id, sentences, lst2, weight_per, context, label_distribution = zip(*data_test) 
            results_test = process_sentences(data_test, sentences, model, is_training=False, learningrate=0.0, config=config, name="test"+str(i),epoch=epoch)
            i += 1

if __name__ == "__main__":
    run_experiment(sys.argv[1])

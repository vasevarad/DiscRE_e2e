import discourseParsing.DiscourseParser as DP
import utils.SenseLabeller as SL
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import csv
import pickle
import time
import math
import numpy as np
import argparse
from tqdm import tqdm


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def embed_discre(message_ids, test_file):
    

    input_dim = 200
    hidden_dim = 200
    seed = 1
    dropout_rate = 0.3
    is_cuda = torch.cuda.is_available()
    grad = 'SGD'
    model_file = './pretrained_models/200_tweet_model.ptstdict'
    word_embedding_dict = './pretrained_models/glove_200.dict'
    split_one_arg = True
    num_direction = 2
    num_layer = 1
    cell_type = 'LSTM'
    attn_act = 'ReLU'
    attn_type = 'element-wise'

    #all the arguments are stored in the opt object
    opt = argparse.Namespace(input_dim=input_dim, hidden_dim=hidden_dim, seed=seed, dropout=dropout_rate, cuda=is_cuda, grad=grad, model=model_file, word_embedding_dict=word_embedding_dict, test_file=test_file, split_one_arg=split_one_arg, num_direction=num_direction, num_layer=num_layer, cell_type=cell_type, attn_act=attn_act, attn_type=attn_type)

    relDump={}
    msgAVGDump={}
    skips=[]

    sl=SL.SenseLabeller()
    
    # fix the seed as '1' for now
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(seed)


    test_word_seqs=pickle.load(open(test_file,"rb"))




    model = DP.DiscourseParser(opt) #change this
    model.load_state_dict(torch.load(model_file))
    if not is_cuda:
        model.to(torch.device("cpu"))
    

    loss_function = nn.BCEWithLogitsLoss()
    # Choose the gradient descent: SGD or Adam

    if is_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()

    total_start=time.time()


    with torch.no_grad():
        if type(test_word_seqs) is list:
            test_ids=range(len(test_word_seqs))
        elif type(test_word_seqs) is dict:
            test_ids=list(test_word_seqs.keys())
        total_start = time.time()

        model.eval()
        start = time.time()
        for id, i in tqdm(zip(message_ids, test_ids)):
            if len(test_word_seqs[i]) < 2:
                if opt.split_one_arg:
                    half=int(len(test_word_seqs[i][0])/2)
                    splitted=[test_word_seqs[i][0][:half],test_word_seqs[i][0][half:]]
                    results = model(('Eval', 'N/A', 0, 1), splitted)
                    if results is None:
                        print("The message at '%d' doesn't have any word in the given embedding dict. Skipping this message"%(i))
                        continue
                    else:
                        class_vec, type_vec, subtype_vec, relation_vec=results
                    relDump[id]={0:torch.cat([class_vec, type_vec, subtype_vec, relation_vec.view(opt.hidden_dim*opt.num_direction*2)]).view(1, -1).cpu().numpy()}
                    msgAVGDump[id]=np.mean(list(relDump[id].values()),axis=0)
                    skips.append(i)
                    continue

                else:
                    skips.append(i)
                    continue
            for j in range(len(test_word_seqs[i])-1):
                results = model(('Eval', 'N/A', j, j+1), test_word_seqs[i])
                if results is None:
                    print("The message at '%d' doesn't have any word in the given embedding dict. Skipping this message"%(i))
                    continue
                else:
                    class_vec, type_vec, subtype_vec, relation_vec=results
                try:
                    relDump[id][j]=torch.cat([class_vec, type_vec, subtype_vec, relation_vec.view(opt.hidden_dim*opt.num_direction*2)]).view(1,-1).cpu().numpy()
                except KeyError:
                    relDump[id]={j:torch.cat([class_vec, type_vec, subtype_vec, relation_vec.view(opt.hidden_dim*opt.num_direction*2)]).view(1, -1).cpu().numpy()}
            if results is not None:
                msgAVGDump[id]=np.mean(list(relDump[id].values()),axis=0)
        end_time = timeSince(start)
        print("Done.")
        if opt.split_one_arg:
            print("Saved at: %s" % ('relDump_SOA_'+opt.model.split('/')[-1]+opt.test_file.split('/')[-1]+'.dict'))
        else:
            print("Saved at: %s" % ('relDump_'+opt.model.split('/')[-1]+opt.test_file.split('/')[-1]+'.dict'))

        print("Prediction Time: %s" % (end_time))
        if len(skips) > 0:
            if opt.split_one_arg:
                print("Warning %i messages were splitted to half because they didn't have more than one discourse argument"%(len(skips)))
            else:
                print("Warning %i messages were skipped because they didn't have more than one discourse argument"%(len(skips)))

            skip_file=open('skipped_or_splitted_ids_'+opt.model.split('/')[-1]+opt.test_file.split('/')[-1]+'.csv','w')
            for idx in skips:
                skip_file.write(str(idx)+'\n')
            skip_file.close()

        if opt.split_one_arg:
            return relDump, msgAVGDump
            pickle.dump(relDump,open('relDump_SOA_'+opt.model.split('/')[-1]+opt.test_file.split('/')[-1]+'.dict','wb'))
            pickle.dump(msgAVGDump, open('avgDump_SOA_' + opt.model.split('/')[-1] + opt.test_file.split('/')[-1] + '.dict', 'wb'))

        else:
            return relDump, msgAVGDump
            pickle.dump(relDump,open('relDump_'+opt.model.split('/')[-1]+opt.test_file.split('/')[-1]+'.dict','wb'))
            pickle.dump(msgAVGDump, open('avgDump_' + opt.model.split('/')[-1] + opt.test_file.split('/')[-1] + '.dict', 'wb'))




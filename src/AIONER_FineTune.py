# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:49:23 2021

@author: luol2
"""


import os
import sys
import argparse
import random
from finetune_model_ner import HUGFACE_NER
from processing_data import ml_intext,out_BIO_BERT_crf
from evaluation_BIO import NER_Evaluation_fn
from tensorflow.keras import callbacks
import tensorflow as tf

gpu = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu))
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)


class NERCallback_PLM(callbacks.Callback):
    def __init__(self, temp_files):
        super(NERCallback_PLM, self).__init__()
        self.tempout = temp_files['infiles']
        self.index_2_label=temp_files['index_2_label']
        self.model_out=temp_files['model_out']
        self.dev_set=temp_files['dev_set']
        self.decoder_type=temp_files['decoder_type']
        
    def on_train_begin(self, logs=None):
        self.max_dev=0.0
        self.max_dev_epoch=0
        self.max_train=0.0
        self.max_train_epoch=0
        self.patein_es=0
    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get("accuracy")
        self.patein_es+=1
        if self.dev_set!=[]:
            print('......dev performance:')
            _dev_predict = self.model.predict(self.dev_set[0])
            #print(_dev_predict)
            if self.decoder_type=='crf':
                out_BIO_BERT_crf(self.tempout['devtemp'],_dev_predict,self.dev_set[1],self.index_2_label)
            elif self.decoder_type=='softmax':
                out_BIO_BERT_softmax(self.tempout['devtemp'],_dev_predict,self.dev_set[1],self.index_2_label)
    
            dev_f1=NER_Evaluation_fn(self.tempout['devtemp'])
            
            if dev_f1>self.max_dev:
                self.max_dev=dev_f1
                self.max_dev_epoch=epoch+1
                self.model.save_weights(self.model_out['BEST'])
        
        if current_acc >self.max_train:
            self.max_train = current_acc
            self.max_train_epoch = epoch+1
            self.model.save_weights(self.model_out['ES'])
            self.patein_es=0
        if self.patein_es>8:
            self.model.stop_training = True
        
        if self.dev_set!=[]:
            print('\nmax_train_acc=',self.max_train,'max_epoch:',self.max_train_epoch,'max_dev_f1=',self.max_dev,'max_epoch:',self.max_dev_epoch,'cur_epoch:',epoch+1)
        else:
            print('\nmax_train_acc=',self.max_train,'max_epoch:',self.max_train_epoch,'lr:',_lr,'cur_epoch:',epoch+1)

   
    
def Hugface_training(infiles,vocabfiles,model_out):
    
    #build model
    plm_model=HUGFACE_NER(vocabfiles)
    plm_model.build_encoder() #PubmedBERT,ELECTRA
    
    plm_model.build_crf_decoder()
    
    #load pre-trained model
    plm_model.load_model(vocabfiles['pretrained'])

    #load dataset
    print('loading dataset......')  
    trainfile=infiles['trainfile']
    train_list = ml_intext(trainfile)
    
    print('numpy dataset......')
    train_x, train_y,train_bert_text_label = plm_model.rep.load_data_hugface(train_list,word_max_len=plm_model.maxlen,label_type='crf') #softmax
    if infiles['devfile']!='':
        devfile=infiles['devfile']
        dev_list = ml_intext(devfile)
        dev_x, dev_y,dev_bert_text_label = plm_model.rep.load_data_hugface(dev_list,word_max_len=plm_model.maxlen,label_type='crf')

    #train model
    if infiles['devfile']!='':
        temp_files={'infiles':infiles,
                    'index_2_label':plm_model.rep.index_2_label,
                    'model_out':model_out,
                    'dev_set':[dev_x,dev_bert_text_label]
                    }
    else:
         temp_files={'infiles':infiles,
                    'index_2_label':plm_model.rep.index_2_label,
                    'model_out':model_out,
                    'dev_set':[]
                    }   
    plm_model.model.fit(train_x,train_y, batch_size=32, epochs=100,verbose=2,callbacks=[NERCallback_PLM(temp_files)])#BioLinkBERT-large: batch_size=16
                             

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='train NER model, python NER_Training.py -t trainfile -d devfile -m modeltype -o outpath')
    parser.add_argument('--trainfile', '-t', help="the training set file",default='../data/conll/ALL_TRAIN.conll')
    parser.add_argument('--devfile', '-d', help="the development set file",default='')
    parser.add_argument('--vocabfile', '-v', help="vocab file with BIO label",default='')
    parser.add_argument('--modeltype', '-m', help="deep learning model (pubmedbert/pubmedbert_full/biolink_base/biolink_large)",default='pubmedbert')   
    parser.add_argument('--outpath', '-o', help="the model output folder",default='../models/')
    args = parser.parse_args()
    if args.outpath[-1]!='/':
        args.outpath+='/'
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
   
    
    infiles={'trainfile':args.trainfile,
             'devfile':args.devfile,
             'devtemp':args.outpath+str(random.randint(10000,50000))+'_tmp_ner.conll',
             }

        
    if args.modeltype=='pubmedbert':
        vocabfiles={'labelfile':args.vocabfile,
                    'checkpoint_path':'../pretrained_models/BiomedNLP-PubMedBERT-base-uncased-abstract/',
                    'lowercase':True,
                    'pretrained':'../models/pubmedbert/pubmedbert-es-AIO.h5'
                    }
            
        model_out={'BEST':args.outpath+'pubmedbert-best-finetune.h5',
                   'ES':args.outpath+'pubmedbert-es-finetune.h5'}
        
    elif args.modeltype=='pubmedbert_full':
        vocabfiles={'labelfile':args.vocabfile,
                    'checkpoint_path':'../pretrained_models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/',
                    'lowercase':True,
                    'pretrained':'../models/pubmedbert_full/pubmedbert_full-es-AIO.h5'
                    }

        model_out={'BEST':args.outpath+'pubmedbert_full-best-finetune.h5',
                   'ES':args.outpath+'pubmedbert_full-es-finetune.h5'}

    elif args.modeltype=='biolink_base':
        vocabfiles={'labelfile':args.vocabfile,
                    'checkpoint_path':'../pretrained_models/BioLinkBERT-base/',
                    'lowercase':True,
                    'pretrained':'../models/biolink_base/biolink_base-es-AIO.h5'
                    }

        model_out={'BEST':args.outpath+'biolink_base-best-finetune.h5',
                   'ES':args.outpath+'biolink_base-es-finetune.h5'}

    elif args.modeltype=='biolink_large':
        vocabfiles={'labelfile':args.vocabfile,
                    'checkpoint_path':'../pretrained_models/BioLinkBERT-large/',
                    'lowercase':True,
                    'pretrained':'../models/biolink_large/biolink_large-es-AIO.h5'
                    }

        model_out={'BEST':args.outpath+'biolink_large-best-finetune.h5',
                   'ES':args.outpath+'biolink_large-es-finetune.h5'}


    Hugface_training(infiles,vocabfiles,model_out)
        
    if os.path.exists(infiles['devtemp']):  #delete tmp file
        os.remove(infiles['devtemp'])
                             

import codecs
import pandas as pd
import os
import subprocess

def read_corpus(filename):
    f = codecs.open(filename, "r", "utf-8")
    corpus = []
    sentence = []
    for line in f:
        if line.strip() == "":
            corpus.append(sentence)
            sentence = []
            continue
        else:
            line = line.strip()
            cline = line.split("\t")
            sentence.append(cline)
    f.close()
    return corpus

def convert_sentence(i,filename, sen):
    new_sen = []
    ind = 1
    working_log_f=open(filename+'_parsing_log3.txt','a')
    working_log_f.write("message number: "+str(i)+"\n")
    working_log_f.close()
    for line in sen:
        working_log_f=open(filename+'_parsing_log3.txt','a')
        working_log_f.write("index: "+str(ind)+"/ line: "+str(line)+'\n')
        working_log_f.close()
        word = line[0]
        if len(line)>1:
            tag = line[1]
        else:
            working_log_f=open(filename+'_error_log3.txt','a')
            working_log_f.write("message number: "+str(i)+"/ index: "+str(ind)+"/ line: "+str(line)+'\n')
            working_log_f.close()
            tag = 'ERROR'
        new_line = [str(ind), word, '_', tag, tag, '_', '0', '_',  '_',   '_']
        new_sen.append(new_line)
        ind += 1
    return new_sen 

def print_sentence(sentence, outputf):
    for line in sentence:
        s = ""                 #unicode for Python2
        for field in line:
            s += field + "\t"   #Python2: unicode + str return unicode
        s = s.strip()           #Python2: Still unicode
        outputf.write(s+"\n")   #Python2: Still unicode
    outputf.write("\n")
    return

def conll_to_output(conll_file):
    corpus = read_corpus(conll_file)
    output_file = conll_file.split("_Tagger")[0] + "_tagger.out"
    outputf = codecs.open(output_file, "w", "utf-8")
    for i, sen in enumerate(corpus):
        new_sen = convert_sentence(i, conll_file, sen)
        print_sentence(new_sen, outputf)
    outputf.close()
    return output_file

def prepare_input_file(csv_file):
    docs = pd.read_csv(csv_file)
    doc_ids = docs['documentID']
    messages = docs['message']
    #remove any new lines into space
    messages = messages.str.replace('\n', ' ')

    # Write messages to a temporary file
    message_file = csv_file.split(".")[0] + "_message_only.csv"
    messages.to_csv(message_file, index=False, header=False)
    return doc_ids, message_file

def tweebo_parse_file(message_file):
    # Construct the command
    print("Jar file location: ",os.path.join(os.getcwd(), 'tweebo_models/ark-tweet-nlp-0.3.2.jar'))
    command = [
        'java', '-XX:ParallelGCThreads=2', '-Xmx500m', '-jar',
        os.path.join(os.getcwd(), 'tweebo_models/ark-tweet-nlp-0.3.2.jar'),
        '--model', os.path.join(os.getcwd(), 'tweebo_models/tagging_model'),
        '--output-format', 'conll', message_file
    ]

    # Run the command and capture the output
    with open(f'{message_file.split(".")[0]}_Tagger_output.txt', 'w') as output_file:
        subprocess.run(command, stdout=output_file)

    # Convert the output to the required format
    output_file = conll_to_output(f'{message_file.split(".")[0]}_Tagger_output.txt')
    return output_file

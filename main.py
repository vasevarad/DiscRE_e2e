from utils.ArgumentExtractor import arg_extractor
from utils.InputMaker import make_discre_input
from utils.TweeboParseUtils import prepare_input_file, tweebo_parse_file
from discourseParsing.EmbedDiscRE import embed_discre
import pickle

def get_discre_embedding(csv_file):
    
    message_ids, message_file = prepare_input_file(csv_file)
    #run tweebo-parser
    output_file = tweebo_parse_file(message_file)
    # Extract the arguments
    args_file, args_meta_file = arg_extractor(output_file)
    # Create the input file
    input_file = make_discre_input(args_file, args_meta_file)
    #get embeddings
    relation_embeddings, average_embeddings = embed_discre(message_ids, input_file)
    return relation_embeddings, average_embeddings
    


if __name__ == '__main__':
    relation_embeddings, average_embeddings = get_discre_embedding('./dummy_texts.csv')
    #save the average embeddings  
    with open('average_embeddings.pkl', 'wb') as f:
        pickle.dump(average_embeddings, f)

    #with open('average_embeddings.pkl', 'rb') as f:
    #    average_embeddings = pickle.load(f)
    #print(average_embeddings.keys())
    #print(average_embeddings['9ddf1293-b1d6-5221-8fac-7e8d78c6d3ab'][0].shape)




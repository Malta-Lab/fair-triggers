from collections import Counter
from pathlib import Path
from argparse import ArgumentParser
from random import shuffle
import string

from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud#, STOPWORDS


STP_ENG = set(stopwords.words('english'))

def clean_string(content):
    content = content.strip()
    content = content.replace('\n', ' ')
    content = content.translate(str.maketrans('', '', string.punctuation))
    content = content.split(' ')
    content = [w.lower() for w in content]
    return content

def export(data):
	words = ""
	for (w,n) in data:
		words += (w + " ") * n	
	
	words = words[:-1]
	words = words.replace(".","").replace(",","").replace("\"","")
	words = words.split()
	shuffle(words)
	return words

def filter_words(words, trigger):
    trigger = [w.lower() for w in trigger.split(' ')]
    print(len(words))
    words = [w for w in words if w[0] not in trigger]
    print(len(words))
    return words

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, default='text_samples.txt')
    parser.add_argument('--output_file', type=str, default='word_cloud.png')
    parser.add_argument('--trigger', type=str, default='')
    args = parser.parse_args()
    
    file_path = Path('./samples') / args.input_file 

    file = open(file_path)

    content = file.read()
    content = clean_string(content)
    word_frequency = Counter(content)
    word_frequency = {k: v for k, v in word_frequency.items() if k not in STP_ENG and v > 1 and k != ''}
    word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True) 
    # TODO: filtering is not working properly
    word_frequency = filter_words(word_frequency, args.trigger)
    print(word_frequency)

    # generate word cloud
    d = word_frequency
    tokens = export(d)
    comment_words = ""
    stopwords = set(STP_ENG)
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)+" "
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='black',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)

    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig(Path('images') / args.output_file)
    plt.show()

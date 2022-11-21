from collections import Counter
from pathlib import Path
from argparse import ArgumentParser
from random import shuffle
import string

from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud#, STOPWORDS

import triggers

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

def filter_words(words, prompt, trigger):
    trigger = [w.lower() for w in trigger]
    print(trigger)
    prompt = [w.lower() for w in prompt.split(' ')]
    print(prompt)
    words = [w for w in words if w[0] not in prompt]
    words = [w for w in words if w[0] not in trigger]
    return words

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trigger', default=None, choices=triggers.trigger_list.keys(), type=str)
    parser.add_argument('--file_path', type=str, default='./samples')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--file_name', type=str, default='trigger_samples.txt')
    parser.add_argument('--words_in_trigger', type=str, nargs='+', default=[])
    parser.add_argument('--model', type=str, default='gpt2')
    
    args = parser.parse_args()

    TRIGGER = args.trigger
    PROMPT = args.prompt
    MODEL = args.model
    file_path = Path(args.file_path)
    file_name = args.file_name

    file = open(file_path / MODEL / TRIGGER / PROMPT.replace(' ','_') / file_name)

    content = file.read()
    content = clean_string(content)
    word_frequency = Counter(content)
    word_frequency = {k: v for k, v in word_frequency.items() if k not in STP_ENG and v > 1 and k != ''}
    word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True) 
    word_frequency = filter_words(word_frequency, PROMPT, args.words_in_trigger)
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
    plt.savefig(file_path / MODEL / TRIGGER / PROMPT.replace(' ','_') / f'{file_name.replace(".txt", ".png")}')
    plt.show()

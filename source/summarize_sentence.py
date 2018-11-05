import re
import nltk
import heapq

class SummarizeParagraph(object):
    """
        Class contains code for text summarization.
    """
    def __init__(self, n):
        # n -> get the n most important summarized sentences.
        self.n = n

    def preprocess(self, text):
         processed_line = re.sub(r'\W+', ' ', text).strip()
         processed_line = re.sub(r'\w*\d\w*', '', processed_line).strip()
         formatted_text = re.sub(r'\s+', ' ', processed_line)  
         return text, formatted_text

    def sent_tokenize(self, text):
        return nltk.sent_tokenize(text)

    def summarize(self, text):
        if not text.strip():
            return ''
        stopwords = nltk.corpus.stopwords.words('english')
        text, preprocessed_text = self.preprocess(text)
        sentence_list = self.sent_tokenize(text)
        word_frequencies = {}  
        for word in nltk.word_tokenize(preprocessed_text):  
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequncy = max(word_frequencies.values())
        for word in word_frequencies.keys():  
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
        
        sentence_scores = {}  
	for sent in sentence_list:  
	    for word in nltk.word_tokenize(sent.lower()):
		if word in word_frequencies.keys():
		    if len(sent.split(' ')) < 70:
		        if sent not in sentence_scores.keys():
		            sentence_scores[sent] = word_frequencies[word]
		        else:
		            sentence_scores[sent] += word_frequencies[word]

        summary_sentences = heapq.nlargest(self.n, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
        return summary


def read_news_articles():
    num_files = 1
    data_str = ''
    for num_file in range(num_files):
        file_path = "Thread_" + str(num_file) + ".dat"
        with open(file_path, "r") as fp:
            data_str += fp.read()
    lines = data_str.split("##")
    lines = map(lambda line: line.split("|"), lines)
    lines =  filter(lambda tup: len(tup) == 2, lines)
    lines = filter(lambda tup: tup[1].strip() != 'Empty', lines)
    lines = map(lambda tup: (tup[0], tup[1].lower().replace("\n", ".")), lines)
    return lines

tuples = read_news_articles()
summarize_para = SummarizeParagraph(8)
for id, text in tuples:
    print "-------------------------------------------------------------------------------------------------------"
    print(summarize_para.summarize(re.sub(r'[^\x00-\x7f]',r'', text)))




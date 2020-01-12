from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import nltk
import string
import csv
import pymysql
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)

conn = pymysql.connect(database='positive_sentiments', user='root' ,password='omshanti123');

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	messages = [line.rstrip() for line in open('train.tsv')]
	for message_no, message in enumerate(messages[:10]):
		print(message_no,message)
		print('\n')

	messages = pd.read_csv('train.tsv', sep='\t',names=["label", "review"])
	messages['length'] = messages['review'].apply(len)
    
	def text_process(mess):
    # Check characters to see if they are in punctuation
		nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
		nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
		return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

	messages['review'].head(5).apply(text_process)
	bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['review'])
	message4 = messages['review']
	bow4 = bow_transformer.transform([message4])
	messages_bow = bow_transformer.transform(messages['review'])
    #print('Shape of Sparse Matrix: ', messages_bow.shape)
	#print('Amount of Non-Zero occurences: ', messages_bow.nnz)
	sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
	tfidf_transformer = TfidfTransformer().fit(messages_bow)
	tfidf4 = tfidf_transformer.transform(bow4)
	messages_tfidf = tfidf_transformer.transform(messages_bow)
	
	spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])		
	all_predictions = spam_detect_model.predict(messages_tfidf)
	msg_train, msg_test, label_train, label_test = \
	train_test_split(messages['review'], messages['label'], test_size=0.2)
	pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
	])
	pipeline.fit(msg_train,label_train)
	

	if request.method == 'POST':
		comment = request.form['comment']
		pro_cat = request.form['name']
		pro_name = request.form['number']

		data = [comment]
		predictions = pipeline.predict(data)
		if predictions == 1:
			#cur = conn.cursor()
			#cur.execute("INSERT INTO positive_sentiments(Product_Category,Product_Name,Review,Sentiment) VALUES(%s,%s,%s,%d)",(pro_cat,pro_name,data,predictions))
			#pymysql.connect.commit()
			#cur.close()
			final_result = pd.DataFrame({'Category':pro_cat,'Name':pro_name,'Review': data,'Sentiment':predictions})
			final_result.to_csv('Negative_Reviews.csv',index=False , mode='a',header=False)
		else:
			final_result = pd.DataFrame({'Category':pro_cat,'Name':pro_name,'Review': data,'Sentiment':predictions})
			final_result.to_csv('Positive_Reviews.csv',index=False , mode='a',header = False)
	return render_template('result.html',prediction = predictions )

if __name__ == '__main__':
	app.run(debug=True,port=7000)

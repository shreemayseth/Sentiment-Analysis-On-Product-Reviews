from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	train_data = pd.read_csv('train_2kmZucJ.csv')
	## importing regular expression library ##
	import re
	def process_tweet(tweet):
		return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", "",tweet.lower()).split())
	train_data['tweet'].head(5).apply(process_tweet)
	x_train, x_test, y_train, y_test = train_test_split(train_data["tweet"],train_data["label"],test_size=0.2,random_state=3)
	count_vect = CountVectorizer(stop_words='english')
	transformer = TfidfTransformer(norm='l2',sublinear_tf=True)
	## for transforming the 80% of the train data ##
	x_train_counts = count_vect.fit_transform(x_train)
	x_train_tfidf = transformer.fit_transform(x_train_counts)
	## for transforming the 20% of the train data which is being used for testing ##
	x_test_counts = count_vect.transform(x_test)
	x_test_tfidf = transformer.transform(x_test_counts)
	from sklearn.ensemble import RandomForestClassifier
	model = RandomForestClassifier(n_estimators=200)
	model.fit(x_train_tfidf,y_train)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		x_test_tfidf = count_vect.transform(data).toarray()
		predictions = model.predict(x_test_tfidf)
	return render_template('result.html',prediction = predictions )

if __name__ == '__main__':
	app.run(debug=True)
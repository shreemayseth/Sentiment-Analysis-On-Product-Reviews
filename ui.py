from flask import Flask,render_template,url_for,request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('amazon.html')


if __name__ == '__main__':
	app.run(debug=True , port=8000)
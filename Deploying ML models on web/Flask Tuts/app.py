from flask import Flask, render_template, redirect, request

# __name__ == __main__
app = Flask(__name__)

friends = ["Prateek", "Jatin", "Sohail"]


num = 5


@app.route('/')
def hello():
	return render_template("index.html", my_friends = friends , number = num )

@app.route('/about')
def about():
	return "<h1> About Page </h1>"


@app.route('/home')
def home():
	return redirect('/')


@app.route('/submit', methods = ['POST'])
def submit_data():
	if request.method == 'POST':
		no1 = int(request.form['no1'])
		no2 = int(request.form['no2'])


		f = request.files['userfile']
		
		f.save(f.filename)


	return str(no1+no2)




if __name__ == '__main__':
	# app.debug = True
	app.run(debug = True)
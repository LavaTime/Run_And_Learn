import os
import socket

import numpy as np
import pandas as pd
from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap

from Baseball_Strike_Zones_SVM.player_params_form import player_params_form
# Constants
from honey_production_linear_regression.predict_form import PredictForm
from viral_tweets_knn.get_tweets_data import get_tweets_data

Work_server = '127.0.0.1'
PORT = 25566

algorithm_id_dict = {'SVC': '0', 'Get_jobs': '1', 'Get_job_counter': '2', 'Get_results': '3', 'Linear_Regression': '4', 'KNN': '5'}


def create_app():
    """
    Create and configure an instance of the Flask application. Connect to the management server.
    :return:
    """
    app = Flask(__name__)
    Bootstrap(app)
    global job_counter
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((Work_server, PORT))
        s.sendall('0|2'.encode())
        job_counter = int(s.recv(1024).decode())
    return app


app = create_app()
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

@app.route('/')
def home():  # put application's code here
    return render_template('home.html')


@app.route('/repos/supervised-learning')
def supervised_learning():
    return render_template('supervised_learning.html')


@app.route('/repos/natural-language-processing')
def natural_language_processing():
    return render_template('natural_language_proccesing.html')


@app.route('/repos/supervised-learning/svm', methods=['GET', 'POST'])
def svm():
    """
    SVM algorithm page, get player name and start and end date. if Post then call server and redirect to status
    :return:
    """
    if request.method == 'GET':
        form = player_params_form()
        form.process()
        return render_template('get_player_params.html', form=form)
    if request.method == 'POST':
        player_name = request.form['player_name']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        call_svm(player_name, start_date, end_date)
        return redirect(url_for('status'))


def call_svm(player_name, start_date, end_date):
    """
    Call server to run SVM algorithm
    :param player_name:
    :param start_date:
    :param end_date:
    :return:
    """
    # call server svm
    global job_counter
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((Work_server, PORT))
        s.sendall(str(job_counter).encode() + '|'.encode() + algorithm_id_dict[
            'SVC'].encode() + '|'.encode() + player_name.encode() + '|'.encode() + start_date.encode() + '|'.encode() + end_date.encode())
    job_counter += 1


@app.route('/repos/supervised-learning/linear-regression', methods=['GET', 'POST'])
def linear_regression():
    """
    Linear Regression algorithm page, get start year and end year. if Post then call server and redirect to status
    :return:
    """
    if request.method == 'GET':
        form = PredictForm()
        form.process()
        return render_template('predict.html', form=form)
    if request.method == 'POST':
        start_year = request.form['start_year']
        end_year = request.form['end_year']
        call_linear_regression(start_year, end_year)
        return redirect(url_for('status'))


def call_linear_regression(start_year, end_year):
    """
    Call server to run Linear Regression algorithm
    :param start_year:
    :param end_year:
    :return:
    """
    # call server linear regression
    global job_counter
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((Work_server, PORT))
        s.sendall(str(job_counter).encode() + '|'.encode() + algorithm_id_dict[
            'Linear_Regression'].encode() + '|'.encode() + start_year.encode() + '|'.encode() + end_year.encode())
    job_counter += 1


@app.route('/repos/natural-language-processing/k-nearest-neighbors', methods=['GET', 'POST'])
def k_nearest_neighbors():
    """
    K-Nearest Neighbors algorithm page, tweet length, followers count, and friends count. if Post then call server and redirect to status
    :return:
    """
    if request.method == 'GET':
        form = get_tweets_data()
        form.process()
        return render_template('get_tweets_data.html', form=form)
    if request.method == 'POST':
        tweet_length = request.form['tweet_length']
        followers_count = request.form['followers_count']
        friends_count = request.form['friends_count']
        call_k_nearest_neighbors(tweet_length, followers_count, friends_count)
        return redirect(url_for('status'))


def call_k_nearest_neighbors(tweet_length, followers_count, friends_count):
    """
    Call server to run K-Nearest Neighbors algorithm
    :param tweet_length:
    :param followers_count:
    :param friends_count:
    :return:
    """
    # call server k nearest neighbors
    global job_counter
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((Work_server, PORT))
        s.sendall(str(job_counter).encode() + '|'.encode() + algorithm_id_dict[
            'KNN'].encode() + '|'.encode() + tweet_length.encode() + '|'.encode() + followers_count.encode() + '|'.encode() + friends_count.encode())
    job_counter += 1


@app.route('/KNN/job/<job_id>', methods=['GET'])
def k_nearest_neighbors_job_view(job_id):
    """
    Check if this job result files are cached, if not cache them and then view the results.
    :param job_id:
    :return:
    """
    if not os.path.exists('./static/jobs/' + job_id):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((Work_server, PORT))
            s.sendall(str(job_id).encode() + '|'.encode() + algorithm_id_dict['Get_results'].encode())
            k_score = get_file(s)
            scatter_3d = get_file(s)
            desc = get_file(s)
            prediction = get_file(s)
            os.mkdir('./static/jobs/' + job_id)
            with open('./static/jobs/' + job_id + '/k_score.png', 'wb') as f:
                f.write(k_score)
                f.close()
            with open('./static/jobs/' + job_id + '/3d_graph.png', 'wb') as f:
                f.write(scatter_3d)
                f.close()
            with open('./static/jobs/' + job_id + '/desc.txt', 'w') as f:
                f.write(desc.decode())
                f.close()
            with open('./static/jobs/' + job_id + '/prediction.txt', 'w') as f:
                f.write(prediction.decode())
                f.close()
    with open('./static/jobs/' + job_id + '/prediction.txt', 'r') as f:
        prediction = f.read()
        f.close()
    return render_template('knn_result.html', job_id=job_id, prediction=prediction)



@app.route('/Linear-Regression/job/<job_id>', methods=['GET'])
def linear_regression_job_view(job_id):
    """
    Check if this job result files are cached, if not cache them and then view the results.
    :param job_id:
    :return:
    """
    if not os.path.exists('./static/jobs/' + job_id):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((Work_server, PORT))
            s.sendall(str(job_id).encode() + '|'.encode() + algorithm_id_dict['Get_results'].encode())
            scatter_img = get_file(s)
            plot_img = get_file(s)
            desc = get_file(s)
            os.mkdir('./static/jobs/' + job_id)
            with open('./static/jobs/' + job_id + '/scatter.png', 'wb') as f:
                f.write(scatter_img)
                f.close()
            with open('./static/jobs/' + job_id + '/plot.png', 'wb') as f:
                f.write(plot_img)
                f.close()
            with open('./static/jobs/' + job_id + '/desc.txt', 'w') as f:
                f.write(desc.decode())
                f.close()
    return render_template('Linear_Regression_result.html', job_id=job_id)




@app.route('/status')
def status():
    """
    get from the server the table of jobs, then if empty return to error page, if not, transform and remove links which are not finished
    then give the table to the render_template function
    :return:
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((Work_server, PORT))
        s.sendall('0|'.encode() + algorithm_id_dict['Get_jobs'].encode())
        jobs = pd.read_json(s.recv(99999).decode())
        if jobs.empty:
            return render_template('error.html', error_message='No jobs'
                                   , link=url_for('home'), link_text='Home')
        jobs.loc[~jobs['Status'].str.contains('finished'), 'Link to informative page'] = 'loading'
        jobs['job_id'] = pd.to_numeric(jobs['job_id'], errors='coerce').fillna(0).astype(np.int64)
        jobs.set_index('job_id', inplace=True)
        jobs.sort_index(inplace=True)
        jobs_html = [jobs.to_html(classes='data', escape=False)]
    return render_template('table_display.html', jobs=jobs_html, titles=jobs.columns.values)


@app.route('/SVC/job/<job_id>', methods=['GET'])
def SVC_job_view(job_id):
    """
    Check if this job result files are cached, if not cache them and then view the results.
    :param job_id:
    :return:
    """
    if not os.path.exists('./static/jobs/' + job_id):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((Work_server, PORT))
            s.sendall(str(job_id).encode() + '|'.encode() + algorithm_id_dict['Get_results'].encode())
            scatter_img = get_file(s)
            plot_img = get_file(s)
            score = get_file(s)
            desc = get_file(s)
            os.mkdir('./static/jobs/' + job_id)
            with open('./static/jobs/' + job_id + '/scatter.png', 'wb') as f:
                f.write(scatter_img)
                f.close()
            with open('./static/jobs/' + job_id + '/plot.png', 'wb') as f:
                f.write(plot_img)
                f.close()
            with open('./static/jobs/' + job_id + '/score.txt', 'w') as f:
                f.write(str(score.decode()))
                f.close()
            with open('./static/jobs/' + job_id + '/desc.txt', 'w') as f:
                f.write(desc.decode())
                f.close()
    with open('./static/jobs/' + job_id + '/score.txt', 'r') as f:
        score = f.read()
        f.close()
        return render_template('SVC_result.html', job_id=job_id, score=score)

@app.after_request
def add_header(r):
    """
    Used for debugging
    :param r:
    :return:
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

def get_file(s):
    """
    get the file from the server by getting the size, and then continuously read the file bytes until finished.
    :param s:
    :return:
    """
    size_bytes = s.recv(1024)
    size = int(size_bytes.decode())
    s.sendall('k'.encode())
    data = b''
    while len(data) < size:
        left_to_read = size - len(data)
        data += s.recv(4096 if left_to_read > 4096 else left_to_read)
    return data

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message='Page not found',
                           link=url_for('home'), link_text='Home')

@app.errorhandler(500)
def internal_server(e):
    return render_template('error.html', error_message='Internal server error',
                           link=url_for('home'), link_text='Home')

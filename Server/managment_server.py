# echo-server
import os
import socket
from os import mkdir
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC

from svm_visualization import draw_boundary
from players import return_player_data

HOST = '127.0.0.1'
PORT = 25566

lock_check = {'id': 'isLocked => Bool'}  # Add dynamic lock_check creation


def update_status(job_id, status):
    while lock_check[job_id]:
        pass
    lock_check[job_id] = True
    with open('Jobs/' + job_id + '/job.status', 'w') as f:
        f.write(status)
        f.close()
    lock_check[job_id] = False


def read_status(job_id):
    print(lock_check)
    while lock_check[job_id]:
        pass
    lock_check[job_id] = True
    with open('Jobs/' + job_id + '/job.status', 'r') as f:
        status = f.read()
        f.close()
    lock_check[job_id] = False
    return status


# t = Thread(target=read_status, args=['0'])
# t.start()

def send_file(s, file):
    s.sendall(str(len(file)).encode())
    s.recv(1024)
    print('Sent:', str(len(file)).encode())
    s.sendall(file)


def do_job(conn, addr):
    print('Connected by', addr)
    while True:
        data = conn.recv(1024).decode()
        if not data:
            break
        print(data)
        split_data = data.split('|')
        job_id, algorithm_id = split_data[:2]
        if algorithm_id == '1':
            print('Get all jobs call received')
            jobs = pd.DataFrame(columns=['job_id', 'Status', 'Link to informative page'])
            jobs_dir = os.fsencode('Jobs/')
            for dir in os.listdir(jobs_dir):
                current_job_id = dir.decode()
                status = read_status(str(current_job_id))
                with open('Jobs/' + current_job_id + '/desc.txt', 'r') as f:
                    desc = f.read()
                    f.close()
                print('<a href=http://77.126.169.99:5000/' + desc + '/job/' + current_job_id + '>' + current_job_id + '</a>')
                jobs = pd.concat([jobs, pd.DataFrame([[current_job_id, status,
                                                       '<a href=http://77.126.169.99:5000/' + desc + '/job/' + current_job_id + '>' + current_job_id + '</a>']],
                                                     columns=['job_id', 'Status', 'Link to informative page'])],
                                 ignore_index=True)

            encoded_jobs = jobs.to_json()
            print(jobs)
            conn.sendall(encoded_jobs.encode())
            print('Get all jobs call finished')
            break
        elif algorithm_id == '2':
            print('Get job counter received')
            d = './Jobs'
            subdirs = list(os.walk(d))[0][1]
            print(subdirs)
            subdirs = list(map(int, subdirs))
            if subdirs == [''] or subdirs == []:
                jobs_count = 0
            else:
                jobs_count = str(max(subdirs) + 1)
                for i in range(int(jobs_count)):
                    print(i)
                    lock_check[str(i)] = False
            conn.sendall(str(jobs_count).encode())
            break
        elif algorithm_id == '3':
            print('Get results')
            with open('Jobs/' + job_id + '/desc.txt', 'r') as f:
                desc = f.read()
                if desc == 'SVC':
                    scatter_img_file = open('Jobs/' + job_id + '/scatter.png', 'rb')
                    scatter_img = scatter_img_file.read()
                    send_file(conn, scatter_img)
                    scatter_img_file.close()
                    plot_img_file = open('Jobs/' + job_id + '/plot.png', 'rb')
                    print('a')
                    plot_img = plot_img_file.read()
                    send_file(conn, plot_img)
                    plot_img_file.close()
                    score_file = open('Jobs/' + job_id + '/score.txt', 'r')
                    score = score_file.read()
                    send_file(conn, score.encode())
                    score_file.close()
                    send_file(conn, desc.encode())
                    print('sent results')
                    break
                if desc == 'Linear-Regression':
                    scatter_img_file = open('Jobs/' + job_id + '/scatter.png', 'rb')
                    scatter_img = scatter_img_file.read()
                    send_file(conn, scatter_img)
                    scatter_img_file.close()
                    plot_img_file = open('Jobs/' + job_id + '/plot.png', 'rb')
                    plot_img = plot_img_file.read()
                    send_file(conn, plot_img)
                    plot_img_file.close()
                    send_file(conn, desc.encode())
                    print('sent results')
                    break
                if desc == 'KNN':
                    k_score_file = open('Jobs/' + job_id + '/k_scores.png', 'rb')
                    k_score = k_score_file.read()
                    send_file(conn, k_score)
                    k_score_file.close()
                    scatter_3d_file = open('Jobs/' + job_id + '/3d_graph.png', 'rb')
                    scatter_3d = scatter_3d_file.read()
                    send_file(conn, scatter_3d)
                    scatter_3d_file.close()
                    send_file(conn, desc.encode())
                    prediction_file = open('Jobs/' + job_id + '/prediction.txt', 'r')
                    prediction = prediction_file.read()
                    send_file(conn, prediction.encode())
                    prediction_file.close()
                    print('sent results')
                    break

        split_data = split_data[2:]
        print('Job ID: ', job_id, 'Received')
        mkdir('Jobs/' + job_id)
        lock_check[job_id] = False
        update_status(job_id, 'Started')
        if algorithm_id == '0':  # SVC
            print('SVC call received')
            update_status(job_id, 'Algorithm found: SVC')  # Status
            with open('Jobs/' + job_id + '/desc.txt', 'w') as f:
                f.write('SVC')
                f.close()
            player_name = split_data[0]
            start_date = split_data[1]
            end_date = split_data[2]
            update_status(job_id, 'SVC: retrieving player data')  # Status
            print(start_date, end_date)
            player_data = return_player_data(player_name, start_date, end_date)
            print(player_data)
            # Add player caching
            update_status(job_id, 'SVC: cleaning data')  # Status
            plt.rcParams['figure.dpi'] = 150  # -------------------------------
            fig, ax = plt.subplots()
            # Map 'S' and 'B' to 1 and 0
            player_data['type'] = player_data['type'].map({'S': 1, 'B': 0})
            # Purge nan(s)
            player_data = player_data.dropna(subset=['plate_x', 'plate_z', 'type'])
            update_status(job_id, 'SVC: plotting scatter')  # Status
            ax.scatter(player_data['plate_x'], player_data['plate_z'], c=player_data['type'], cmap=plt.cm.coolwarm,
                        alpha=0.25)
            ax.set_xlabel('Distance from plate - X axis (m)')
            ax.set_ylabel('Distance from plate - Z axis (m)')
            ax.set_title('Strike or Ball of ' + player_name)
            update_status(job_id, 'SVC: splitting data')  # Status
            training_set, validation_set = train_test_split(player_data)
            # ---------------------
            GAMMA, C = 1000, 1000
            # ---------------------
            update_status(job_id, 'SVC: training model')  # Status
            classifier = SVC(kernel='rbf', gamma=GAMMA, C=C)
            classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
            # Draw the boundary
            update_status(job_id, 'SVC: drawing boundary')  # Status
            fig.savefig('Jobs/' + job_id + '/scatter.png')
            draw_boundary(ax, classifier)
            update_status(job_id, 'SVC: calculating accuracy')  # Status
            # Find accuracy of our algorithm
            score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])
            update_status(job_id, 'SVC: saving plot and score')  # Status
            fig.savefig('Jobs/' + job_id + '/plot.png')
            #ax.clf()
            with open('Jobs/' + job_id + '/score.txt', 'w') as f:
                f.write(str(score))
                f.close()
            update_status(job_id, 'SVC: finished')  # Status
        elif algorithm_id == '4':  # Linear regression
            print('Linear regression call received')
            update_status(job_id, 'Algorithm found: Linear Regression')  # Status
            with open('Jobs/' + job_id + '/desc.txt', 'w') as f:
                f.write('Linear-Regression')
                f.close()
            start_year = split_data[0]
            end_year = split_data[1]
            update_status(job_id, 'Linear Regression: Loading dataset')  # Status
            dataset = pd.read_csv('datasets/honeyproduction.csv')
            update_status(job_id, 'Linear Regression: cleaning data')  # Status
            plt.rcParams['figure.dpi'] = 100  # -------------------------------
            fig, ax = plt.subplots()
            production_per_year = dataset.groupby('year').totalprod.mean().reset_index()
            X_values = production_per_year['year']
            # Reshape the data to fit
            X_values = X_values.values.reshape(-1, 1)
            Y_values = production_per_year['totalprod']
            update_status(job_id, 'Linear Regression: plotting scatter')  # Status
            ax.scatter(X_values, Y_values)
            ax.set_xlabel('Year')
            ax.set_ylabel('Honey Production (kg)')
            ax.set_title('Honey Production per Year')
            update_status(job_id, 'Linear Regression: training model')  # Status
            regressor = LinearRegression()
            regressor.fit(X_values, Y_values)
            update_status(job_id, 'Linear Regression: plotting model')  # Status
            y_predict = regressor.predict(X_values)
            ax.plot(X_values, y_predict)
            update_status(job_id, 'Linear Regression: saving plot')  # Status
            fig.savefig('Jobs/' + job_id + '/scatter.png')
            update_status(job_id, 'Linear Regression: predicting future')  # Status
            # Calculate the future using the model
            end_year, start_year = (max((end_year, start_year)), min((end_year, start_year)))
            X_future = np.array(range(int(start_year), int(end_year)))
            X_future = X_future.reshape(-1, 1)
            future_predict = regressor.predict(X_future)
            ax.plot(X_future, future_predict, color='orange')
            update_status(job_id, 'Linear Regression: saving final plot')  # Status
            fig.savefig('Jobs/' + job_id + '/plot.png')
            # ax.clf()
            update_status(job_id, 'Linear Regression: finished')  # Status
        elif algorithm_id == '5':  # KNN
            print('KNN call received')
            update_status(job_id, 'Algorithm found: KNN')  # Status
            with open('Jobs/' + job_id + '/desc.txt', 'w') as f:
                f.write('KNN')
                f.close()
            tweet_length = split_data[0]
            followers_count = split_data[1]
            friends_count = split_data[2]
            update_status(job_id, 'KNN: loading tweets dataset')  # Status
            all_tweets = pd.read_json('datasets/random_tweets.json', lines=True)
            update_status(job_id, 'KNN: cleaning and transforming data')  # Status
            median_retweets = all_tweets['retweet_count'].median()
            all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] >= median_retweets, 1, 0)
            all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
            all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
            all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)
            labels = all_tweets['is_viral']
            data = all_tweets[['tweet_length', 'followers_count', 'friends_count']]
            scaled_data = scale(data, axis=0)
            plt.rcParams['figure.dpi'] = 100  # -------------------------------
            fig, ax = plt.subplots()
            update_status(job_id, 'KNN: splitting data')  # Status
            training_data, validation_data, training_labels, validation_labels = train_test_split(scaled_data, labels, test_size=0.2)
            update_status(job_id, 'KNN: training and testing models')  # Status
            scores = []
            max_score = -1
            best_classifier = None
            for k in range(1, 200):
                classifier = KNeighborsClassifier(n_neighbors=k)
                classifier.fit(training_data, training_labels)
                scores.append(classifier.score(validation_data, validation_labels))
                if scores[-1] > max_score:
                    best_classifier = classifier
                    max_score = scores[-1]
            update_status(job_id, 'KNN: plot k-score graph')  # Status
            ax.plot(range(1, 200), scores)
            ax.set_xlabel('"k" value')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('K-Nearest-Neighbor Accuracy to "k"')
            fig.savefig('Jobs/' + job_id + '/k_scores.png')
            with open('Jobs/' + job_id + '/score.txt', 'w') as f:
                f.write(str(max(scores)))
                f.close()
            update_status(job_id, 'KNN: predicting for entered values')  # Status
            prediction = 'Viral' if best_classifier.predict([[tweet_length, followers_count, friends_count]]) else 'Not Viral'
            with open('Jobs/' + job_id + '/prediction.txt', 'w') as f:
                f.write(prediction)
                f.close()
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            x_line = all_tweets['tweet_length']
            y_line = all_tweets['followers_count']
            z_line = all_tweets['friends_count']
            ax.scatter3D(x_line, y_line, z_line, c=all_tweets['is_viral'], cmap='seismic')
            ax.set_xlabel('Tweet Length')
            ax.set_ylabel('Followers Count')
            ax.set_zlabel('Friends Count')
            fig.savefig('Jobs/' + job_id + '/3d_graph.png')
            update_status(job_id, 'KNN: finished')  # Status



threads = []

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        threads.append(Thread(target=do_job, args=(conn, addr)))
        threads[-1].start()

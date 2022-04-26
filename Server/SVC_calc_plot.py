import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from Server.svm_visualization import draw_boundary
import pandas as pd
# from players import return_player_data


def return_fig(raw_player_data):
    #plt.rcParams['figure.dpi'] = 300

    fig, ax = plt.subplots()
    # print(player_data.description.unique())
    player_data = pd.DataFrame(raw_player_data, columns=['type', 'plate_x', 'plate_z'])
    print(player_data.head())
    # Map 'S' and 'B' to 1 and 0
    player_data['type'] = player_data['type'].map({'S': 1, 'B': 0})
    # print(player_data['type'])
    # print(player_data['plate_x'])
    # Purge nan(s)
    player_data = player_data.dropna(subset=['plate_x', 'plate_z', 'type'])
    plt.scatter(player_data['plate_x'], player_data['plate_z'], c=player_data['type'], cmap=plt.cm.coolwarm, alpha=0.25)
    # print(player_data.describe())
    # SVC
    training_set, validation_set = train_test_split(player_data, random_state=1)  # Remember to remove random_state

    # ---------------------
    GAMMA, C = 1, 1
    # ---------------------
    classifier = SVC(kernel='rbf', gamma=GAMMA, C=C)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
    # Draw the boundry

    draw_boundary(ax, classifier)
    # Find accuracy of our algorithm
    score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])
    return fig, score


def calc_classifier_score(gamma, c):
    itering_classifier = SVC(kernel='rbf', gamma=gamma, C=c)
    itering_classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
    return itering_classifier, (
        itering_classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))

# iters = 10
# maximum_score = 0
# for current_gamma in range(1, iters):
#   for current_c in range(1, iters):
#     current_classifer, current_score = calc_classifier_score(current_gamma, current_c)
#     if current_score > maximum_score:
#       maximum_score = current_score
#       best_gamma, best_c, max_classifier = current_gamma, current_c, current_classifer
# print(maximum_score, best_gamma, best_c)
# plt.scatter(player_data['plate_x'], player_data['plate_z'], c=player_data['type'], cmap=plt.cm.coolwarm, alpha=0.25)
# draw_boundary(ax, max_classifier)
# plt.show()

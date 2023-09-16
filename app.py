# # Importing essential libraries
# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np
# import pandas as pd
# import xgboost
# from xgboost import XGBRegressor
# from flask_cors import CORS, cross_origin
# # Load the Ridge Regression Classifier model
# # change the model type to see lil changes in prediction
# # filename = 'odi_score_predict_model_xgb_2.1_more_venues.pkl'
# filename = 'odi_score_predict_model_xgb_2.1_more_venues.pkl'
# pipe = pickle.load(open(filename, 'rb'))



# app = Flask(__name__)
# CORS(app, support_credentials=True)
# @cross_origin(supports_credentials=True)


# @app.route('/')
# def home():
#     return jsonify({'data': 'API'})


# @app.route('/', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         batting_team = request.json['batting_team']
#         bowing_team = request.json['bowling_team']
#         venue = request.json['venue']
#         current_score = int(request.json['runs'])
#         wickets = int(request.json['wickets'])
#         overs = float(request.json['overs'])
#         runs_in_prev_5 = int(request.json['runs_in_prev_5'])

        
#         balls_left = 300 - overs*6
#         wickets_left = 10 - wickets
#         crr = current_score/overs
#         score_prev_5 = current_score - runs_in_prev_5

#     input_df = pd.DataFrame(
#     {'bat_team': [batting_team], 'bowl_team': [bowing_team], 'venue':[venue], 'runs': [current_score], 'wickets_left': [wickets_left], 'runrate': [crr], 'balls_left': [balls_left], 'runs_last_5': [score_prev_5]})

#     result = pipe.predict(input_df)

#     return jsonify({"predict": str(int(result[0]))})
# if __name__ == '__main__':
#     app.run(debug=True)


# Importing essential libraries
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
from flask_cors import CORS, cross_origin

# filename = 'odi_score_predict_model_xgb_2.1_more_venues.pkl'
filename = 'pipe_xgb.pkl'
# filename = 'pipe_randomforest.pkl'
pipe = pickle.load(open(filename, 'rb'))





app = Flask(__name__)
CORS(app, support_credentials=True)
@cross_origin(supports_credentials=True)


@app.route('/')
def home():
    return jsonify({'data': 'API'})
 

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        batting_team = request.json['batting_team']
        bowing_team = request.json['bowling_team']
        venue = request.json['venue']
        current_score = int(request.json['runs'])
        wickets = int(request.json['wickets'])
        wickets_last_5 = int(request.json['wickets_last_5'])
        overs = float(request.json['overs'])
        runs_in_prev_5 = int(request.json['runs_in_prev_5'])

        
        # balls_left = 300 - overs*6
        wickets_left = 10 - wickets
        crr = current_score/overs
        score_prev_5 = current_score - runs_in_prev_5
        wickets_prev_5 = wickets - wickets_last_5
        
        remaining_overs = 49.6 - overs
        weight_overs = remaining_overs / 49.6
        weight_wicket = wickets_left / 10
        merge_weight = (remaining_overs * weight_overs) + (wickets_left*weight_wicket)
    input_df = pd.DataFrame(
    {'venue':[venue],'bat_team': [batting_team], 'bowl_team': [bowing_team], 'overs': [overs], 'runs': [current_score], 'wickets': [wickets], 'runrate': [crr], 'runs_last_5': [score_prev_5], 'wickets_last_5': [wickets_prev_5], 'merge_weight': [merge_weight]})

    result = pipe.predict(input_df)

    return jsonify({"predict": str(int(result[0]))})
if __name__ == '__main__':
    app.run(debug=True)

import glob
import os
import pathlib
from io import BytesIO
import flask
from werkzeug.utils import secure_filename
import PredictionModelling as PM
import pslanalysis as psl
from flask import request, jsonify, flash, send_file
from flask import Flask, flash, request, redirect, url_for

UPLOAD_FOLDER = './testfile'
ALLOWED_EXTENSIONS = {'m4a'}

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def home():
    return "<h1>Welcome to PSL Cricket Analysis Api</p>"

@app.route('/topwinnerteam', methods=['GET'])
def topwinner():
    topwinnerteam = PM.PMF()
    return jsonify(topwinnerteam)

@app.route('/top_players', methods=['GET'])
def top_players_request():
    top_players, byseasonwinnerteam, topwinnerteam=psl.psl_analysis()
    return jsonify(top_players)

@app.route('/byseasonwinnerteam', methods=['GET'])
def byseasonwinnerteam_request():
    top_players, byseasonwinnerteam, topwinnerteam=psl.psl_analysis()
    return jsonify(byseasonwinnerteam)

@app.route('/topwinnerteamf', methods=['GET'])
def topwinnerteam_request():
    top_players, byseasonwinnerteam, topwinnerteam=psl.psl_analysis()
    return jsonify(topwinnerteam)




if __name__ == '__main__':
    app.run(debug=True)

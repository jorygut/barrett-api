#Import requirements
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS, cross_origin
from flask import current_app
from flask.helpers import send_from_directory
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from statsmodels.api import add_constant
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import xml.etree.ElementTree as ET
import cv2
from flask import send_file
from flask_vite import Vite
from io import BytesIO
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import werkzeug
from werkzeug.utils import secure_filename
import asyncio
import librosa

#Configure Flask API
app = Flask(__name__)
vite = Vite(app)
CORS(app)  # Allow all origins for simplicity
print('app_started')

#Create upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Create home route
@app.route("/")
def index():
    return {'test': 'hello'}

#Create xml route
@app.route("/api/xml", methods=['POST'])

def read_xml_file(file):
    # Parse xml file
    tree = ET.parse(file)
    root = tree.getroot()
    
    # Configure tracking data
    part_count = 0
    detection_count = 0
    tracks_info = {
        'nTracks': int(root.attrib['nTracks']),
        'spaceUnits': root.attrib['spaceUnits'],
        'frameInterval': float(root.attrib['frameInterval']),
        'timeUnits': root.attrib['timeUnits'],
        'generationDateTime': root.attrib['generationDateTime'],
        'from': root.attrib['from']
    }

    particle_data = []
    for particle in root.findall('./particle'):
        part_count += 1
        particle_info = {'nSpots': int(particle.attrib['nSpots'])}
        if particle_info['nSpots'] < 100:
            continue

        detections = []
        previous_detection = None
        for detection in particle.findall('./detection'):
            detection_count += 1
            detection_info = {
                't': int(detection.attrib['t']),
                'x': float(detection.attrib['x']),
                'y': float(detection.attrib['y']),
                'z': float(detection.attrib['z']),
                'speed': 0
            }
            if previous_detection is not None:
                delta_x = detection_info['x'] - previous_detection['x']
                delta_y = detection_info['y'] - previous_detection['y']
                
                time_elapsed = tracks_info['frameInterval'] * (detection_info['t'] - previous_detection['t'])
                
                detection_info['speed'] = math.sqrt(delta_x ** 2 + delta_y ** 2) / time_elapsed
            
            detections.append(detection_info)
            previous_detection = detection_info
        
        particle_info['detections'] = detections
        particle_data.append(particle_info)

    tracks_info['particles'] = particle_data
    return tracks_info


#Detect feed lawns
def detect_lighter_circles(image_path, par1, par2, lawn_count):
    #Read and grayscale image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Apply gaussian blur and detect circle
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=2, minDist=100,
                               param1=par1, param2=par2, minRadius=10,
                               maxRadius=100)

    circle_data = []
    x_cords = {}
    y_cords = {}
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        #Check lawn accuracy
        if len(circles) == lawn_count:
            for (x, y, r) in circles:
                circle_coordinates = []
                for i in range(x - r, x + r + 1):
                    for j in range(y - r, y + r + 1):
                        if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                            circle_coordinates.append((i, j))
                            x_cords[i] = 1
                            y_cords[j] = 1
                circle_data.append({
                    'center_x': x,
                    'center_y': y,
                    #'radius': r,
                    'coordinates': circle_coordinates
                })
        else:
            x_cords = 'fail'
            y_cords = 'fail'
    return x_cords,y_cords
#Find correct lawns
def configure_circle(img,lawn_count):
    if img:
        flag = False
        for param1 in range(20, 55):
            print('configuring')
            if flag:
                break
            for param2 in range(20, 55):
                x_cords, y_cords = detect_lighter_circles(img, param1, param2,lawn_count) 
                if x_cords != 'fail':
                    flag = True
                    break  
    return x_cords,y_cords
#Create output csv
def create_file(input_file,lawn_count, img):
    print('creating file')
    if lawn_count > 0 and img != 0:
        x_cords, y_cords = configure_circle(img,lawn_count)
    print('circle configured')
    tracks_info = read_xml_file(input_file)
    print(tracks_info)
    particle_count = 0
    detection_count = 0

    part_df = pd.DataFrame()
    total_speed = 0
    for i in tracks_info['particles']:
        detection_count = 0
        particle_count += 1
        speed1 = 0
        speed2 = 0
        speed3 = 0
        #Build csv and calculate variables
        for j in i['detections']:
            detection_count += 1
            seconds = (j['t'] * 1.24324324324)
            total_speed += j['speed']
            if j['t'] < 29:
                speed1 += j['speed']
            if j['t'] < 90.3:
                speed2 += j['speed']
            if j['t'] < 121:
                speed3 += j['speed']
            if lawn_count > 0:
                if (int(j['x'] + 1) in x_cords or int(j['x'] - 1) in x_cords) and (int(j['y'] + 1) in y_cords or int(j['y'] - 1) in y_cords):
                    part_df.at[detection_count,f'obs_{particle_count}_in_lawn'] = True
                else:
                    part_df.at[detection_count,f'obs_{particle_count}_in_lawn'] = False
                    if (seconds <= 90):
                        part_df.at[detection_count,f'obs_{particle_count}_left_lawn_90_secs'] = True
            part_df.at[detection_count,f'obs_{particle_count}_x'] = j['x']
            part_df.at[detection_count,f'obs_{particle_count}_y'] = j['y']
            part_df.at[detection_count,f'obs_{particle_count}_t'] = j['t']
            part_df.at[detection_count,f'obs_{particle_count}_z'] = j['z']
            part_df.at[detection_count,f'obs_{particle_count}_speed'] = j['speed']
            part_df.at[detection_count, f'obs_{particle_count}_time_seconds'] = seconds
            part_df.at[detection_count, f'obs_{particle_count}_average_speed_before_shock'] = (speed1 / 29)
            part_df.at[detection_count, f'obs_{particle_count}_average_speed_during_shock'] = (speed2 / 61)
            part_df.at[detection_count, f'obs_{particle_count}_average_speed_after_shock'] = (speed2 / 31)
    average_speed = (total_speed / 230) / detection_count
    print(part_df)
    return part_df
#Calculate distance between detections
def calculate_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
#Prepare dataframe for regression model
def analyze(df):
    for i in range(1, 31): 
        obs_x_col = f'obs_{i}_x'
        obs_y_col = f'obs_{i}_y'
        total_distance_col = f'obs_{i}_total_distance'
        obs_speed_col = f'obs_{i}_speed'
        average_speed_col = f'obs_{i}_average_speed'
        leaves_lawn_col = f'obs_{i}_in_lawn'

        if obs_x_col in df.columns and obs_y_col in df.columns:
            total_distance = 0
            prev_x = df.iloc[0][obs_x_col]
            prev_y = df.iloc[0][obs_y_col]

            for index, row in df.iterrows():
                if pd.notnull(row[obs_x_col]) and pd.notnull(row[obs_y_col]):
                    current_x = row[obs_x_col]
                    current_y = row[obs_y_col]
                    total_distance += calculate_distance(prev_x, prev_y, current_x, current_y)
                    prev_x, prev_y = current_x, current_y
            total_speed = 0
            num_speeds = 0  # Counter for non-null speed values
            if obs_speed_col in df.columns:  # Check if speed column exists
                for j in df[obs_speed_col]:
                    if pd.notnull(j):  # Check if speed value is not null
                        total_speed += j
                        num_speeds += 1
            
            if num_speeds > 0:  # Check if there are non-null speed values
                average_speed = total_speed / num_speeds
                df[average_speed_col] = average_speed
            
            df[total_distance_col] = total_distance
    return df
#Create new dataframe for regression model
def make_reg_df(df, strain):
    lis = []
    for i in range(1, 31): 
        obs_total_distance = f'obs_{i}_total_distance'
        obs_average_speed = f'obs_{i}_average_speed'
        obs_lawn_col = f'obs_{i}_in_lawn'
        obs_speed_before = f'obs_{i}_average_speed_before_shock'
        obs_speed_during = f'obs_{i}_average_speed_during_shock'
        obs_speed_after = f'obs_{i}_average_speed_after_shock'

        if obs_total_distance in df.columns and obs_average_speed in df.columns:
            leaves_lawn = False
            if obs_lawn_col in df.columns:
                if not df[obs_lawn_col].all(): 
                    leaves_lawn = True
            worm_stats = {
                'Strain': strain,
                'Obs_Number': i,
                'Total_Distance': df[obs_total_distance].iloc[0],  # Assuming all rows in an observation have the same total distance
                'Average_Speed': df[obs_average_speed].iloc[0],  # Assuming all rows in an observation have the same average speed
                'Leaves_Lawn': leaves_lawn,
                'Average_Speed_Before_Shock': df[obs_speed_before].iloc[200],
                'Average_Speed_During_Shock': df[obs_speed_during].iloc[200],
                'Average_Speed_After_Shock': df[obs_speed_after].iloc[200],
            }
            lis.append(worm_stats)
    return lis
#Create OLS regression model
def regress(file, x_data, y_data):
    #Encode string data
    label_encoder = LabelEncoder()
    x_vars = x_data.split(',')
    y_vars = y_data.split(',')   
    #Ensure variables are floats
    df = file
    df = df.fillna(0)
    df['Strain'] = label_encoder.fit_transform(df['Strain'])
    df['Leaves_Lawn'] = df['Leaves_Lawn'].astype(float)
    df['Strain'] = df['Strain'].astype(float)
    df['Total_Distance'] = pd.to_numeric(df['Total_Distance'], errors='coerce')

    X = sm.add_constant(df[x_vars])
    y = df[y_vars]

    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("VIF Data:")
    print(vif_data)

    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    #Graph model
    for x_var in x_vars:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[x_var], y=y[y_vars[0]])
        sns.lineplot(x=df[x_var], y=results.fittedvalues, color='red')
        plt.title(f'Scatter plot of {x_var} vs {y_vars[0]} with Regression Line')
        plt.xlabel(x_var)
        plt.ylabel(y_vars[0])
        plt.savefig(f'scatter_plot_{x_var}_vs_{y_vars[0]}.png')
        plt.close()

    plt.figure(figsize=(10, 6))
    sns.residplot(x=results.fittedvalues, y=results.resid, lowess=True, line_kws={'color': 'red', 'lw': 1})
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.savefig('residuals_vs_fitted.png')
    plt.close()

    fig = plt.figure(figsize=(12, 8))
    sm.graphics.plot_partregress_grid(results, fig=fig)
    plt.savefig('partial_regression_plot.png')
    plt.close()

    return results
#Get x and y variables
def extract_regression_variables(data):
    selected_x = data.get('selectedX', [])
    selected_y = data.get('selectedY', [])
    return selected_x, selected_y

def predict_tracks(track_file):
    df = pd.DataFrame()
    tracks = pd.read_csv(track_file)
    
    # Iterate through files in 'RegressionFiles' directory
    for file in os.listdir('/RegressionFiles'):
        if file.endswith('.csv') and 'distance' not in file: 
            name = file.split()
            strain = name[1]
            cur_df = pd.read_csv(os.path.join('/RegressionFiles', file))
            # Rename columns with strain prefix
            for col in cur_df.columns:
                new_col_name = f'{strain}_{col}'
                cur_df = cur_df.rename(columns={col: new_col_name})
            df = pd.concat([df, cur_df.reset_index(drop=True)], axis=1)
    
    #Preprocessing
    df = df.fillna(0)
    df = df.loc[:, ~df.columns.str.contains('unnamed', case=False)]

    target_columns = [col for col in df.columns if 'x' in col or 'y' in col]
    features = [col for col in df.columns if col not in target_columns]

    X = df[features]
    y = df[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    next_coordinates = model.predict(df[features])

    fig, ax = plt.subplots(figsize=(10, 5))

    actual_x = df[[col for col in df.columns if 'x' in col]]
    actual_y = df[[col for col in df.columns if 'y' in col]]

    #Plot actual values
    ax.scatter(actual_x, actual_y, color='blue', label='Actual', s=5)

    #Plot predicted values
    predicted_x = next_coordinates[:, ::2]  
    predicted_y = next_coordinates[:, 1::2]  
    ax.scatter(predicted_x, predicted_y, color='red', label='Predicted', s=5)  

    ax.set_title('Actual vs Predicted Values')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.grid(True)

    filename = 'predicted_tracks.png'
    fig.savefig(filename, format='png')

    return fig
#Create error bars
def make_error_bars(df, x, y):
    grouped = df.groupby(x)[y].agg(['mean', 'std'])

    x_values = grouped.index
    y_means = grouped['mean'].values
    y_std = grouped['std'].values

    plt.errorbar(x_values, y_means, yerr=y_std, fmt='o', capsize=5, label='Data with error bars')

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'Plot with Error Bars ({x} vs {y})')
    plt.legend()
    plt.grid(True)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.clf()

    return buffer

def analyze_patterns(df):
    values = []
    strain_dic = {}
    speed_dic = {}
    lawn_dic = {}
    speed_increase_dict = {}
    speed_decrease_dict = {}
    distance_dict = {}
    for index, row in df.iterrows():
        if row['Strain'] not in strain_dic:
            strain_dic[row['Strain']] = 1
            speed_dic[row['Strain']] = row['Average_Speed']
            speed_increase_dict[row['Strain']] = ((row['Average_Speed_During_Shock'] - row['Average_Speed_Before_Shock']) / row['Average_Speed_Before_Shock']) * 100
            speed_decrease_dict[row['Strain']] = ((row['Average_Speed_After_Shock'] - row['Average_Speed_During_Shock']) / row['Average_Speed_During_Shock']) * 100
            if row['Leaves_Lawn'] == True:
                lawn_dic[row['Strain']] = 1
            else:
                lawn_dic[row['Strain']] = 0
        else:
            strain_dic[row['Strain']] += 1
            speed_dic[row['Strain']] += row['Average_Speed']
            if row['Leaves_Lawn'] == True:
                lawn_dic[row['Strain']] += 1
            if row['Average_Speed_Before_Shock'] != 0:
                speed_increase_dict[row['Strain']] += ((row['Average_Speed_During_Shock'] - row['Average_Speed_Before_Shock']) / row['Average_Speed_Before_Shock']) * 100
            else:
                speed_increase_dict[row['Strain']] += 0
            if row['Average_Speed_During_Shock'] != 0:
                speed_decrease_dict[row['Strain']] += ((row['Average_Speed_After_Shock'] - row['Average_Speed_During_Shock']) / row['Average_Speed_During_Shock']) * 100
            else:
                speed_decrease_dict[row['Strain']] += 0
        distance_dict[row['Strain']] += row['Total_Distance']
    for i in strain_dic:
        data = {
            'strain': i,
            'count': strain_dic[i],
            'average_speed': speed_dic[i] / strain_dic[i],
            'percent_leaves_lawn': lawn_dic[i] / strain_dic[i],
            'average_speed_shock_start': (speed_increase_dict[i] / strain_dic[i]),
            'average_speed_shock_end': speed_decrease_dict[i] / strain_dic[i],
            'total_distance': distance_dict[i]
        }
        values.append(data)
    return values
def analyze_music(file_path):
    try:
        # Load the audio file directly with librosa
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract tempo (beats per minute)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        
        # Extract chroma features and estimate the key
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = chroma.mean(axis=1).argmax()
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_name = keys[key]
        
        # Harmonic-Percussive Source Separation to isolate the harmonic part
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Extract bass line from the harmonic part
        bass_chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        bass = np.mean(bass_chroma[:3], axis=0)
        bass_note = bass.argmax() % len(keys)
        bass_note_name = keys[bass_note]
        
        # Convert the bass note to Hz
        A4 = 440
        C0 = A4 * 2**(-4.75)
        bass_frequency = C0 * 2**(bass_note / 12.0)
        
        # Extract additional features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        rms = np.mean(librosa.feature.rms(y=y))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        hnr = np.mean(librosa.effects.harmonic(y))
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strength = np.mean(onset_env)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr), axis=1)
        
        # Create a dictionary with the extracted values
        results = {
            "tempo": tempo,
            "key": key_name,
            "bass_note": bass_note_name,
            "bass_frequency": bass_frequency,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_contrast": spectral_contrast,
            "spectral_flatness": spectral_flatness,
            "rms_energy": rms,
            "zero_crossing_rate": zero_crossing_rate,
            "harmonic_to_noise_ratio": hnr,
            "beat_strength": beat_strength,
            "tonnetz": tonnetz
        }
        
        return results
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
#API call for image, lawn count, and xml file
@app.route('/image', methods=['POST'])
def upload_image_and_number():
    print('image uploaded')

    if 'xml_file' not in request.files:
        return jsonify({"error": "No XML file part in the request"}), 400
    #Get data from API
    image_file = request.files.get('image_file')
    number = request.form.get('number')
    xml_file = request.files.get('xml_file')
    print(xml_file)
    print(number)
    print(image_file)

    uploads_dir = os.path.join(app.config['UPLOAD_FOLDER'])
    os.makedirs(uploads_dir, exist_ok=True) 

    #Handle image
    if (image_file and image_file.filename.endswith(('.jpg', '.jpeg', '.png'))):

        image_filename = secure_filename(image_file.filename)

        image_path = os.path.join(uploads_dir, image_filename)

        image_file.save(image_path)
    else:
        image_path = 0
    print('starting df')
    try:
        number = int(number)
    except ValueError:
        return jsonify({"error": "Number is not valid"}), 400
    df = create_file(xml_file, number, image_path)
    print('finished_df')
    print(df)
    xml_filename = os.path.splitext(xml_file.filename)[0]

    #Handle csvs
    csv_filename = f"{xml_filename}.csv"
    print(csv_filename)

    #Save df
    df.to_csv(os.path.join(uploads_dir, csv_filename), index=False)

    #Send file
    return send_file(os.path.join(uploads_dir, csv_filename), as_attachment=True)
#Create and download regression model
@app.route("/regress", methods=['POST'])
def perform_regression():
    final_df = pd.DataFrame()
    uploaded_files =  request.files.getlist('csv')
    x_data = request.form.get('xdata')
    y_data = request.form.get('ydata')
    print(uploaded_files)
    strain = ""
    for file in uploaded_files:
        title = str(file).split()
        strain = title[2]
        df = pd.read_csv(file)
        df = analyze(df)
        f = make_reg_df(df,strain)
        final_df = pd.concat([final_df, pd.DataFrame(f)], ignore_index=True)
    print(final_df)
    results = regress(final_df, x_data, y_data)
    with open("Regression.txt", "w") as file:
        file.write(results.summary().as_text())
    return send_file("Regression.txt", as_attachment=True)
#Download graphs
@app.route('/partreg', methods=['GET'])
def download_png():
    return send_file('partial_regression_plot.png', as_attachment=True)
@app.route('/resfit', methods=['GET'])
def res_vs_fit():
    return send_file('residuals_vs_fitted.png', as_attachment=True)

@app.route('/ml', methods=['POST', 'GET'])
def build_model():
    if 'model' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
    files = request.files.getlist('model')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400
    
    predicted_tracks = predict_tracks(files[0])
    
    return send_file('predicted_tracks.png', as_attachment=True)
@app.route('/graph', methods=['POST', 'GET'])
def send_error_bars():
    final_df = pd.DataFrame()
    uploaded_files =  request.files.getlist('csv')
    x_data = request.form.get('xdata')
    y_data = request.form.get('ydata')
    print(uploaded_files)
    strain = ""
    for file in uploaded_files:
        title = str(file).split()
        strain = title[2]
        df = pd.read_csv(file)
        df = analyze(df)
        f = make_reg_df(df,strain)
        final_df = pd.concat([final_df, pd.DataFrame(f)], ignore_index=True)
    print(final_df)
    png = make_error_bars(final_df, x_data, y_data)
    with open('error_bars.png', 'wb') as f:
        f.write(png.read())
    return send_file('error_bars.png', as_attachment=True)

@app.route('/analyze', methods=['POST', 'GET'])
def get_insight():
    final_df = pd.DataFrame()
    uploaded_files =  request.files.getlist('files')
    print(uploaded_files)
    strain = ""
    for file in uploaded_files:
        title = str(file).split()
        strain = title[2]
        df = pd.read_csv(file)
        df = analyze(df)
        f = make_reg_df(df,strain)
        final_df = pd.concat([final_df, pd.DataFrame(f)], ignore_index=True)
    print(final_df)
    final_df.fillna(0, inplace=True)
    all_strains = analyze_patterns(final_df)
    print(all_strains)
    return jsonify(all_strains)

@app.route('/audio', methods=['POST', 'GET'])
def analyze_mp3():
    song = request.files.get('mp3')
    print(song)
    results = analyze_music(song)
    return jsonify(results)
#Api test route
@app.route('/test',  methods=['POST', 'GET'])
def test():
    return {'test': 'hello'}
#Run main and create upload folder
if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run()


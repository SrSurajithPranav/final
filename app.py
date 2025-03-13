import os
import random
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from flask_caching import Cache
from functools import lru_cache
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from gtts import gTTS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from wordcloud import WordCloud
import base64
from apscheduler.schedulers.background import BackgroundScheduler
from pyod.models.iforest import IForest
from flask_babel import Babel, _
from flask_babel import Babel

app = Flask(__name__)
app.config['BABEL_DEFAULT_LOCALE'] = 'en'  # Set default locale
babel = Babel(app)

@babel.localeselector
def get_locale():
    return 'en'  # Modify this to dynamically detect locale if needed

app.jinja_env.globals['get_locale'] = get_locale  # Make it available in Jinja templates

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['BABEL_DEFAULT_LOCALE'] = 'en'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
babel = Babel(app)

# Cache setup
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Twitter API setup
API_ENDPOINT = "https://api.twitter.com/2/users/by/username/"
BEARER_TOKENS = [
    "AAAAAAAAAAAAAAAAAAAAALOhywEAAAAAW8Oi86wzl4ft4tnhzRlyZ3%2FFGF8%3D4ItRbnSYTeK9jcWopAugYeMcqfAOypNi5gBERQ4wBjY4aq9phL",
    "AAAAAAAAAAAAAAAAAAAAAANoxAEAAAAARIratdNtUpsn7Gxk5YZHrDgXVmI%3DhdjZY09cKTCe7xAioFXli8PM2qq68rtGjVcqFwYAvGjlnAARsY",
    "AAAAAAAAAAAAAAAAAAAAACg7zAEAAAAABBikdwYlorE2FeNpNqrkT8uV1fk%3DsxxU1YZAYwO0G56tjTPSElgal0CEy0HF3zJ4a5jtpmTdRyO4h7",
    "AAAAAAAAAAAAAAAAAAAAACzpygEAAAAA8k18d8ZP23NtWodqYI5x6mwfS58%3DDLK7sv0qrqEu7u7bovNoHegux5EkHiVhKqp41jPV1mKzYRcQMm"
]
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class MonitoredUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    username = db.Column(db.String(80), nullable=False)
    last_followers = db.Column(db.Integer, nullable=True)
    last_checked = db.Column(db.DateTime, default=datetime.utcnow)

class SharedReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    username = db.Column(db.String(80))
    prediction = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    notes = db.Column(db.Text)
    shared_at = db.Column(db.DateTime, default=datetime.utcnow)

class GameScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    score = db.Column(db.Integer)
    achieved_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@babel.localeselector
def get_locale():
    return request.accept_languages.best_match(['en', 'es', 'fr'])

@lru_cache(maxsize=100)
def get_twitter_data_cached(username):
    params = {"user.fields": "public_metrics,created_at,description,verified,profile_image_url"}
    for token in BEARER_TOKENS:
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.get(f"{API_ENDPOINT}{username}", headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if 'Rate limit' in str(e):
                return {'error': 'Rate limit exceeded'}
            return None
    return None

def parse_twitter_data(user_data):
    user = user_data['data']
    return pd.DataFrame([{
        'followers_count': user['public_metrics']['followers_count'],
        'friends_count': user['public_metrics']['following_count'],
        'statuses_count': user['public_metrics']['tweet_count'],
        'listed_count': user['public_metrics']['listed_count']
    }])

def simulate_live_data():
    return {
        'bot_detections': np.random.randint(100, 500),
        'active_users': np.random.randint(50, 200),
        'suspicious_activity': np.random.randint(10, 100),
        'world_map_data': [
            {'lat': np.random.uniform(-90, 90), 'lon': np.random.uniform(-180, 180), 'value': np.random.randint(1, 10)}
            for _ in range(20)
        ],
        'network_data': {
            'nodes': [{'id': str(i), 'group': np.random.randint(1, 5)} for i in range(10)],
            'links': [{'source': str(i), 'target': str((i+1)%10), 'value': np.random.randint(1, 5)} for i in range(10)]
        }
    }

class Analyzer:
    def __init__(self):
        self.static_data = self.load_static_data()
        self.model, self.scaler = self.train_model()

    def load_static_data(self):
        users = pd.read_csv('static_data/users.csv')
        fusers = pd.read_csv('static_data/fusers.csv')
        users['label'] = 0
        fusers['label'] = 1
        data = pd.concat([users, fusers], ignore_index=True)
        return data

    def train_model(self):
        X = self.static_data[['followers_count', 'friends_count', 'statuses_count', 'listed_count']]
        y = self.static_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)
        print(classification_report(y_test, model.predict(X_test)))
        return model, scaler

    def predict(self, user_data, use_lightweight=False):
        features = user_data[['followers_count', 'friends_count', 'statuses_count', 'listed_count']]
        features = self.scaler.transform(features)
        prediction = self.model.predict(features)[0]
        confidence = self.model.predict_proba(features)[0][prediction]
        return 'fake' if prediction == 1 else 'genuine', confidence

analyzer = Analyzer()

def monitor_user(monitored_user_id):
    with app.app_context():
        monitored_user = MonitoredUser.query.get(monitored_user_id)
        if not monitored_user:
            return
        username = monitored_user.username
        user_data = get_twitter_data_cached(username)
        if user_data and 'data' in user_data:
            current_followers = user_data['data']['public_metrics']['followers_count']
            if monitored_user.last_followers is not None:
                change = ((current_followers - monitored_user.last_followers) / monitored_user.last_followers) * 100
                if abs(change) > 10:
                    print(f"Alert: {username} followers changed by {change:.2f}%")
            monitored_user.last_followers = current_followers
            monitored_user.last_checked = datetime.utcnow()
            db.session.commit()

scheduler = BackgroundScheduler()
scheduler.start()

@app.route('/')
def index():
    static_users = analyzer.static_data['name'].tolist()
    return render_template('index.html', static_users=static_users)

@app.route('/analyze', methods=['POST'])
def analyze():
    username = request.form['username'].strip().lstrip('@')
    if not username:
        return render_template('error.html', error=_("Please enter a username")), 400

    use_lightweight = 'lightweight' in request.form
    user_data = get_twitter_data_cached(username)
    if user_data and 'data' in user_data:
        user_df = parse_twitter_data(user_data)
        prediction, confidence = analyzer.predict(user_df, use_lightweight)
        features = user_df.to_dict(orient='records')[0]
        profile_image = user_data['data'].get('profile_image_url', '')
        description = user_data['data'].get('description', '')
        created_at = user_data['data'].get('created_at', '')
        verified = user_data['data'].get('verified', False)
    else:
        static_data = analyzer.static_data[analyzer.static_data['name'] == username]
        if not static_data.empty:
            prediction, confidence = analyzer.predict(static_data, use_lightweight)
            features = static_data[['followers_count', 'friends_count', 'statuses_count', 'listed_count']].to_dict(orient='records')[0]
            profile_image = ''
            description = 'Static user data'
            created_at = 'N/A'
            verified = False
        else:
            return render_template('error.html', error=_("User not found or API limit reached")), 404

    clf = IForest(contamination=0.1, random_state=42)
    features_df = pd.DataFrame([features])
    clf.fit(features_df)
    anomaly_prediction = clf.predict(features_df)[0]
    anomaly_score = clf.decision_scores_[0]

    behavior = {
        'timestamps': [str(datetime.now() - timedelta(days=i)) for i in range(7)],
        'activity': [random.randint(0, 100) for _ in range(7)]
    }

    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(description) if description else {'compound': 0}
    sentiment_history = [
        sia.polarity_scores(f"Simulated post {i} for {username}")['compound']
        for i in range(5)
    ]
    sentiment_timestamps = [(datetime.now() - timedelta(days=4-i)).strftime('%Y-%m-%d') for i in range(5)]

    wordcloud_data = Counter(description.split()).most_common(10) if description else [('No', 1), ('description', 1)]
    wordcloud = WordCloud(width=400, height=200, background_color='black').generate_from_frequencies(dict(wordcloud_data))
    wordcloud_image = BytesIO()
    wordcloud.to_image().save(wordcloud_image, format='PNG')
    wordcloud_base64 = base64.b64encode(wordcloud_image.getvalue()).decode('utf-8')

    result = {
        'username': username,
        'prediction': prediction,
        'confidence': confidence,
        'features': features,
        'behavior': behavior,
        'sentiment': sentiment,
        'sentiment_history': {
            'values': sentiment_history,
            'timestamps': sentiment_timestamps
        },
        'wordcloud': wordcloud_base64,
        'profile_image': profile_image,
        'description': description,
        'created_at': created_at,
        'verified': verified,
        'anomaly': 'Anomaly Detected' if anomaly_prediction == 1 else 'No Anomaly',
        'anomaly_score': anomaly_score
    }
    return render_template('result.html', result=result)

@app.route('/download/<format>/<username>')
@login_required
def download(format, username):
    prediction = request.args.get('prediction', 'N/A')
    confidence = request.args.get('confidence', 'N/A')
    result = {
        'username': username,
        'prediction': prediction,
        'confidence': float(confidence),
        'timestamp': datetime.utcnow().isoformat()
    }

    if format == 'pdf':
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"Analysis Report for @{username}", styles['Title']),
            Paragraph(f"Prediction: {prediction}", styles['Normal']),
            Paragraph(f"Confidence: {confidence}", styles['Normal'])
        ]
        doc.build(story)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f"{username}_report.pdf")
    
    elif format == 'json':
        buffer = BytesIO()
        buffer.write(json.dumps(result, indent=4).encode('utf-8'))
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f"{username}_report.json")
    
    elif format == 'excel':
        df = pd.DataFrame([result])
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f"{username}_report.xlsx")
    
    return jsonify({'error': _('Invalid format')}), 400

@app.route('/voice_summary/<username>')
@login_required
def voice_summary(username):
    user_data = get_twitter_data_cached(username)
    if user_data and 'data' in user_data:
        prediction = request.args.get('prediction', 'N/A')
        confidence = request.args.get('confidence', 'N/A')
        text = f"Analysis for {username}. Prediction: {prediction}. Confidence: {confidence}."
        tts = gTTS(text=text, lang='en')
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return send_file(audio_file, mimetype='audio/mpeg', as_attachment=True, download_name=f"{username}_summary.mp3")
    return jsonify({'error': _('User data not found')}), 404

@app.route('/live_dashboard')
def live_dashboard():
    live_data = simulate_live_data()
    return render_template('live_dashboard.html', live_data=live_data)

@app.route('/control_room')
def control_room():
    live_data = simulate_live_data()
    return render_template('control_room.html', live_data=live_data)

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for('index'))
        flash(_('Invalid credentials'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    data = request.json
    username = data['username']
    label = 1 if data['is_fake'] else 0

    user_data = get_twitter_data_cached(username)
    if user_data and 'data' in user_data:
        user_df = parse_twitter_data(user_data)
    else:
        static_data = analyzer.static_data[analyzer.static_data['name'] == username]
        if not static_data.empty:
            user_df = static_data
        else:
            return jsonify({'status': 'error', 'message': _('User data not found.')}), 404

    user_df['label'] = label
    analyzer.static_data = pd.concat([analyzer.static_data, user_df], ignore_index=True)
    analyzer.model, analyzer.scaler = analyzer.train_model()

    return jsonify({'status': 'success', 'message': _('Feedback recorded and model updated.')})

@app.route('/monitor/<username>', methods=['POST'])
@login_required
def add_to_monitoring(username):
    monitored_user = MonitoredUser.query.filter_by(user_id=current_user.id, username=username).first()
    if not monitored_user:
        monitored_user = MonitoredUser(user_id=current_user.id, username=username)
        db.session.add(monitored_user)
        db.session.commit()
        scheduler.add_job(
            monitor_user,
            'interval',
            minutes=5,
            args=[monitored_user.id],
            id=f'monitor_{monitored_user.id}'
        )
    return jsonify({'status': 'success', 'message': _(f'{username} added to monitoring')})

@app.route('/stop_monitor/<username>', methods=['POST'])
@login_required
def stop_monitoring(username):
    monitored_user = MonitoredUser.query.filter_by(user_id=current_user.id, username=username).first()
    if monitored_user:
        scheduler.remove_job(f'monitor_{monitored_user.id}')
        db.session.delete(monitored_user)
        db.session.commit()
    return jsonify({'status': 'success', 'message': _(f'{username} removed from monitoring')})

@app.route('/save_score', methods=['POST'])
@login_required
def save_score():
    score = request.json['score']
    game_score = GameScore(user_id=current_user.id, score=score)
    db.session.add(game_score)
    db.session.commit()
    return jsonify({'status': 'success', 'message': _('Score saved')})

@app.route('/leaderboard')
def leaderboard():
    scores = GameScore.query.order_by(GameScore.score.desc()).limit(10).all()
    return render_template('leaderboard.html', scores=scores)

@app.route('/share', methods=['POST'])
@login_required
def share():
    username = request.form['username']
    prediction = request.form['prediction']
    confidence = float(request.form['confidence'])
    notes = request.form['notes']
    shared_report = SharedReport(
        user_id=current_user.id,
        username=username,
        prediction=prediction,
        confidence=confidence,
        notes=notes
    )
    db.session.add(shared_report)
    db.session.commit()
    return redirect(url_for('shared_reports'))

@app.route('/shared_reports')
def shared_reports():
    reports = SharedReport.query.all()
    return render_template('shared_reports.html', reports=reports)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
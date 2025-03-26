from flask import Flask, render_template, request, redirect, url_for, make_response
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import tweepy
import os
from io import BytesIO
import base64
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
from textblob import TextBlob
import folium
from folium.plugins import HeatMap
import json
import io

app = Flask(__name__)

# Twitter API setup with four bearer tokens for load balancing
BEARER_TOKENS = [
    os.getenv('TWITTER_BEARER_TOKEN_1', 'YOUR_TOKEN_1_HERE'),
    os.getenv('TWITTER_BEARER_TOKEN_2', 'YOUR_TOKEN_2_HERE'),
    os.getenv('TWITTER_BEARER_TOKEN_3', 'YOUR_TOKEN_3_HERE'),
    os.getenv('TWITTER_BEARER_TOKEN_4', 'YOUR_TOKEN_4_HERE')
]
current_token_index = 0

def get_twitter_client():
    global current_token_index
    token = BEARER_TOKENS[current_token_index]
    current_token_index = (current_token_index + 1) % len(BEARER_TOKENS)
    return tweepy.Client(bearer_token=token)

class Analyzer:
    def __init__(self):
        self.static_data = self.load_static_data()
        self.model = self.train_model()
        self.explainer = shap.TreeExplainer(self.model)

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
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X, y)
        print("Best parameters:", grid_search.best_params_)
        print("Classification Report:\n", classification_report(y, grid_search.predict(X)))
        return grid_search.best_estimator_

    def analyze_user(self, username):
        try:
            client = get_twitter_client()
            user = client.get_user(username=username.replace('@', ''), user_fields=['public_metrics', 'description', 'location', 'verified'])
            if not user.data:
                return None

            profile = {
                'followers_count': user.data.public_metrics['followers_count'],
                'friends_count': user.data.public_metrics['following_count'],
                'statuses_count': user.data.public_metrics['tweet_count'],
                'listed_count': user.data.public_metrics['listed_count'],
                'description': user.data.description or 'No description',
                'location': user.data.location or None,
                'verified': user.data.verified
            }

            X_new = pd.DataFrame([profile])[['followers_count', 'friends_count', 'statuses_count', 'listed_count']]
            prediction = self.model.predict(X_new)[0]
            confidence = self.model.predict_proba(X_new)[0][prediction]

            # 1. Confidence Score Visualization (Heat Map)
            confidence_heatmap = BytesIO()
            plt.figure(figsize=(2, 2))
            sns.heatmap([[confidence]], cmap='RdYlGn', annot=True, fmt='.2f', cbar=False)
            plt.axis('off')
            plt.savefig(confidence_heatmap, format='png', bbox_inches='tight')
            confidence_heatmap.seek(0)
            confidence_heatmap_base64 = base64.b64encode(confidence_heatmap.getvalue()).decode('utf-8')
            plt.close()

            # 2. Feature Contribution Breakdown (SHAP Plot)
            shap_values = self.explainer.shap_values(X_new)
            shap_plot = BytesIO()
            plt.figure(figsize=(6, 4))
            shap.summary_plot(shap_values, X_new, plot_type='bar', show=False)
            plt.savefig(shap_plot, format='png', bbox_inches='tight')
            shap_plot.seek(0)
            shap_plot_base64 = base64.b64encode(shap_plot.getvalue()).decode('utf-8')
            plt.close()

            # Waterfall Chart (using Plotly)
            shap_values_single = shap_values[prediction][0]
            features = ['followers_count', 'friends_count', 'statuses_count', 'listed_count']
            waterfall_data = {
                'Feature': features,
                'Contribution': shap_values_single
            }
            fig = px.bar(waterfall_data, x='Contribution', y='Feature', orientation='h', title='Feature Contribution to Prediction')
            waterfall_plot = fig.to_json()

            # 3. Similar Account Analysis (Cluster Graph - Simplified)
            distances = np.linalg.norm(self.static_data[features].values - X_new.values, axis=1)
            similar_indices = distances.argsort()[:3]
            similar_accounts = self.static_data.iloc[similar_indices]
            cluster_data = {
                'nodes': [{'id': username, 'group': 'target'}] + 
                        [{'id': f"similar_{i}", 'group': 'similar'} for i in range(len(similar_indices))],
                'links': [{'source': username, 'target': f"similar_{i}", 'value': 1} for i in range(len(similar_indices))]
            }

            # 4. Sentiment Analysis (Pie Chart)
            sentiment = TextBlob(profile['description']).sentiment
            polarity = sentiment.polarity
            sentiment_labels = ['Positive', 'Neutral', 'Negative']
            sentiment_values = [
                max(0, polarity),  # Positive
                max(0, 1 - abs(polarity)),  # Neutral
                max(0, -polarity)  # Negative
            ]
            sentiment_data = {'labels': sentiment_labels, 'values': sentiment_values}

            # Word Cloud
            wordcloud_data = Counter(profile['description'].split()).most_common(10)
            wordcloud = WordCloud(width=400, height=200, background_color='black').generate_from_frequencies(dict(wordcloud_data))
            wordcloud_image = BytesIO()
            wordcloud.to_image().save(wordcloud_image, format='PNG')
            wordcloud_base64 = base64.b64encode(wordcloud_image.getvalue()).decode('utf-8')

            # 5. Anomaly Detection (Distribution Plot)
            anomaly_scores = np.abs(X_new.values - self.static_data[features].mean().values) / self.static_data[features].std().values
            anomaly_score = anomaly_scores.mean()
            anomaly_plot = BytesIO()
            plt.figure(figsize=(6, 4))
            sns.histplot(anomaly_scores.flatten(), kde=True, color='orange')
            plt.axvline(anomaly_score, color='red', linestyle='--', label=f'Account Anomaly Score: {anomaly_score:.2f}')
            plt.legend()
            plt.title('Anomaly Score Distribution')
            plt.savefig(anomaly_plot, format='png', bbox_inches='tight')
            anomaly_plot.seek(0)
            anomaly_plot_base64 = base64.b64encode(anomaly_plot.getvalue()).decode('utf-8')
            plt.close()

            # Timeline Chart (Placeholder)
            timeline_data = {'dates': ['2023-01', '2023-02', '2023-03'], 'scores': [0.5, 0.6, anomaly_score]}

            # 6. Geolocation Visualization (Interactive Map)
            map_html = None
            if profile['location']:
                m = folium.Map(location=[0, 0], zoom_start=2)
                folium.Marker([0, 0], popup=profile['location']).add_to(m)  # Mock coordinates
                map_html = m._repr_html_()

            # Heatmap (Placeholder)
            heatmap_data = [[0, 0, 1]]  # Mock data

            # 7. Network Analysis (Social Graph - Simplified)
            network_data = {
                'nodes': [{'id': username, 'group': 'target'}] + 
                        [{'id': f"friend_{i}", 'group': 'friend'} for i in range(min(3, profile['friends_count']))],
                'links': [{'source': username, 'target': f"friend_{i}", 'value': 1} for i in range(min(3, profile['friends_count']))]
            }

            # 8. Profile Image Analysis (Placeholder)
            profile_image_analysis = "Profile image analysis not implemented (requires image processing)."

            # 9. Verification and Trust Indicator
            trust_score = confidence * 0.8 + (0.2 if profile['verified'] else 0)

            return {
                'username': username,
                'prediction': 'fake' if prediction == 1 else 'genuine',
                'confidence': confidence,
                'confidence_heatmap': confidence_heatmap_base64,
                'shap_plot': shap_plot_base64,
                'waterfall_plot': waterfall_plot,
                'cluster_data': cluster_data,
                'sentiment_data': sentiment_data,
                'wordcloud': wordcloud_base64,
                'anomaly_plot': anomaly_plot_base64,
                'timeline_data': timeline_data,
                'map_html': map_html,
                'heatmap_data': heatmap_data,
                'network_data': network_data,
                'profile_image_analysis': profile_image_analysis,
                'verified': profile['verified'],
                'trust_score': trust_score,
                'profile': profile
            }
        except Exception as e:
            print(f"Error analyzing {username}: {e}")
            return None

analyzer = Analyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form.get('username')
        if username:
            return redirect(url_for('result', username=username))
    return render_template('index.html')

@app.route('/result/<username>')
def result(username):
    result = analyzer.analyze_user(username)
    if not result:
        return "User not found or API error.", 404
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

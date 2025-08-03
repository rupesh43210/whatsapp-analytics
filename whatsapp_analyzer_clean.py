#!/usr/bin/env python3
"""
Clean & Optimized WhatsApp Chat Analytics Tool
Fixed colors, orientations, and blank charts
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from collections import Counter, defaultdict
import argparse
import os
from textblob import TextBlob
import emoji
from wordcloud import WordCloud
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

class CleanWhatsAppAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.messages = []
        self.df = None
        self.participants = set()
        self.clean_names = {}
        
    def clean_participant_name(self, name):
        """Clean and standardize participant names"""
        if name in self.clean_names:
            return self.clean_names[name]
        
        # Remove common suffixes and prefixes
        cleaned = re.sub(r'@.*?(?=\s|$)', '', name)  # Remove @mentions
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Remove extra spaces
        
        # Extract first name and apartment/unit if available
        parts = cleaned.split()
        if len(parts) > 1:
            # Try to identify apartment numbers or units
            for part in parts:
                if re.match(r'[A-Z]-?\d+|[A-Z]\d+|\d+[A-Z]?', part):
                    first_name = parts[0]
                    unit = part
                    cleaned = f"{first_name} ({unit})"
                    break
            else:
                # Just use first and last name
                if len(parts) >= 2:
                    cleaned = f"{parts[0]} {parts[-1]}"
                else:
                    cleaned = parts[0]
        
        self.clean_names[name] = cleaned
        return cleaned
    
    def parse_messages(self):
        """Enhanced parser for various WhatsApp export formats"""
        patterns = [
            # Format: [DD/MM/YY, HH:MM:SS AM/PM] Name: Message
            r'\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}:\d{2})\s+(AM|PM)\]\s*([^:]+?):\s*(.*)',
            # Format: [DD/MM/YY, HH:MM:SS] Name: Message
            r'\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}:\d{2})\]\s*([^:]+?):\s*(.*)',
            # Format: DD/MM/YY, HH:MM - Name: Message
            r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2})\s*[-–]\s*([^:]+?):\s*(.*)',
        ]
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        current_message = None
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            message_found = False
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if current_message:
                        self.messages.append(current_message)
                    
                    groups = match.groups()
                    if len(groups) == 4:
                        date_str, time_str, sender, message = groups
                    elif len(groups) == 5:
                        if groups[2] in ['AM', 'PM']:
                            date_str, time_str, am_pm, sender, message = groups
                            time_str = f"{time_str} {am_pm}"
                        else:
                            date_str, time_str, sender, am_pm, message = groups
                            time_str = f"{time_str} {am_pm}"
                    
                    # Skip system messages
                    if message.strip().startswith('‎') or 'added' in message or 'created this group' in message:
                        continue
                    
                    clean_sender = self.clean_participant_name(sender.strip())
                    
                    current_message = {
                        'line_number': line_num,
                        'date_str': date_str,
                        'time_str': time_str,
                        'sender': clean_sender,
                        'original_sender': sender.strip(),
                        'message': message.strip()
                    }
                    self.participants.add(clean_sender)
                    message_found = True
                    break
            
            if not message_found and current_message:
                current_message['message'] += ' ' + line
        
        if current_message:
            self.messages.append(current_message)
        
        self._process_timestamps()
        print(f"📊 Parsed {len(self.messages)} messages from {len(self.participants)} participants")
        self._create_dataframe()
        
    def _process_timestamps(self):
        """Process various timestamp formats with better handling"""
        for msg in self.messages:
            date_str = msg['date_str']
            time_str = msg['time_str']
            
            date_formats = [
                '%d/%m/%y', '%d/%m/%Y', '%m/%d/%y', '%m/%d/%Y',
                '%d.%m.%y', '%d.%m.%Y', '%Y-%m-%d'
            ]
            
            time_formats = [
                '%H:%M:%S %p', '%I:%M:%S %p', '%H:%M:%S', '%H:%M', '%I:%M %p'
            ]
            
            timestamp = None
            for date_fmt in date_formats:
                for time_fmt in time_formats:
                    try:
                        timestamp = datetime.strptime(f"{date_str} {time_str}", f"{date_fmt} {time_fmt}")
                        break
                    except ValueError:
                        continue
                if timestamp:
                    break
            
            if not timestamp:
                try:
                    timestamp = datetime.strptime(f"{date_str} {time_str[:8]}", '%d/%m/%y %H:%M:%S')
                except ValueError:
                    timestamp = datetime.now()
            
            msg['timestamp'] = timestamp
            msg['date'] = timestamp.date()
            msg['time'] = timestamp.time()
            msg['hour'] = timestamp.hour
            msg['day_of_week'] = timestamp.strftime('%A')
            msg['month'] = timestamp.strftime('%B')
            msg['year'] = timestamp.year
            msg['week_of_year'] = timestamp.isocalendar()[1]
    
    def _create_dataframe(self):
        """Create enhanced pandas DataFrame with additional features"""
        if not self.messages:
            raise ValueError("No messages found. Please check the file format.")
        
        self.df = pd.DataFrame(self.messages)
        self.df['word_count'] = self.df['message'].apply(lambda x: len(x.split()))
        self.df['char_count'] = self.df['message'].apply(len)
        self.df['emoji_count'] = self.df['message'].apply(lambda x: len([c for c in x if c in emoji.EMOJI_DATA]))
        self.df['is_media'] = self.df['message'].str.contains(r'<Media omitted>|<attached:|image omitted|video omitted|audio omitted|document omitted', case=False, na=False)
        self.df['is_deleted'] = self.df['message'].str.contains(r'This message was deleted|You deleted this message', case=False, na=False)
        self.df['is_question'] = self.df['message'].str.contains(r'\?', na=False)
        self.df['has_url'] = self.df['message'].str.contains(r'http[s]?://|www\.', case=False, na=False)
        
        # Add time-based features
        self.df['date_formatted'] = self.df['timestamp'].dt.strftime('%B %d, %Y')
        self.df['time_period'] = pd.cut(self.df['hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['Night (12-6AM)', 'Morning (6-12PM)', 'Afternoon (12-6PM)', 'Evening (6-12AM)'],
                                       include_lowest=True)
    
    def create_clean_visualizations(self):
        """Create clean, properly oriented visualizations with correct colors"""
        
        # Create a clean dashboard with proper spacing
        fig = make_subplots(
            rows=6, cols=2,
            subplot_titles=[
                '📊 Messages by Participant (Descending Order)', '⏰ Activity by Hour (24-hour format)',
                '📅 Daily Activity Pattern', '📈 Monthly Message Trends',
                '💭 Average Words per Message', '🎯 Content Types',
                '🕐 Hourly Activity Heatmap', '🗓️ Daily Message Volume',
                '😊 Sentiment Distribution', '📱 Time Period Activity',
                '🌟 Top 15 Words', '😀 Top 10 Emojis'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.12
        )
        
        # Define consistent color palette
        colors_main = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                      '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                      '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D5A6BD',
                      '#AED6F1', '#A9DFBF']
        
        # 1. Messages by Participant (FIXED - Proper vertical bars, correct colors)
        participant_counts = self.df['sender'].value_counts().head(15)  # Top 15 for better visibility
        
        fig.add_trace(
            go.Bar(
                x=participant_counts.index,
                y=participant_counts.values,
                name='Messages Count',
                marker=dict(
                    color=colors_main[:len(participant_counts)],
                    line=dict(color='rgba(0,0,0,0.3)', width=1)
                ),
                text=[f"{v:,}" for v in participant_counts.values],
                textposition='outside',
                textfont=dict(size=10, color='black'),
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Hourly Activity (FIXED - Proper colors and labels)
        hourly_data = self.df['hour'].value_counts().sort_index()
        hour_labels = []
        for h in hourly_data.index:
            if h == 0:
                hour_labels.append('12 AM')
            elif h < 12:
                hour_labels.append(f'{h} AM')
            elif h == 12:
                hour_labels.append('12 PM')
            else:
                hour_labels.append(f'{h-12} PM')
        
        # Create color gradient for hours (darker = more messages)
        max_val = hourly_data.max()
        hour_colors = [f'rgba(65, 182, 196, {0.3 + 0.7 * (val/max_val)})' for val in hourly_data.values]
        
        fig.add_trace(
            go.Bar(
                x=hour_labels,
                y=hourly_data.values,
                name='Hourly Messages',
                marker=dict(color=hour_colors, line=dict(color='rgba(0,0,0,0.3)', width=1)),
                text=[f"{v:,}" for v in hourly_data.values],
                textposition='outside',
                textfont=dict(size=9, color='black'),
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Daily Activity Pattern (FIXED)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_data = self.df['day_of_week'].value_counts().reindex(days_order, fill_value=0)
        
        # Color gradient for days
        max_day = daily_data.max()
        day_colors = [f'rgba(76, 209, 196, {0.3 + 0.7 * (val/max_day)})' for val in daily_data.values]
        
        fig.add_trace(
            go.Bar(
                x=daily_data.index,
                y=daily_data.values,
                name='Daily Messages',
                marker=dict(color=day_colors, line=dict(color='rgba(0,0,0,0.3)', width=1)),
                text=[f"{v:,}" for v in daily_data.values],
                textposition='outside',
                textfont=dict(size=10, color='black'),
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Monthly Trends (FIXED - Real data)
        monthly_data = self.df.groupby([self.df['timestamp'].dt.year, self.df['timestamp'].dt.month]).size()
        month_labels = []
        month_values = []
        
        for (year, month), count in monthly_data.items():
            month_name = datetime(year, month, 1).strftime('%b %Y')
            month_labels.append(month_name)
            month_values.append(count)
        
        fig.add_trace(
            go.Scatter(
                x=month_labels,
                y=month_values,
                mode='lines+markers+text',
                name='Monthly Trend',
                line=dict(color='#FF6B6B', width=4),
                marker=dict(size=12, color='#FF6B6B', line=dict(color='white', width=2)),
                text=[f"{v:,}" for v in month_values],
                textposition='top center',
                textfont=dict(size=10, color='black'),
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Average Words per Message (FIXED)
        words_by_participant = self.df.groupby('sender')['word_count'].mean().sort_values(ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(
                x=words_by_participant.index,
                y=words_by_participant.values,
                name='Avg Words',
                marker=dict(color='#96CEB4', line=dict(color='rgba(0,0,0,0.3)', width=1)),
                text=[f"{v:.1f}" for v in words_by_participant.values],
                textposition='outside',
                textfont=dict(size=9, color='black'),
                hovertemplate='<b>%{x}</b><br>Avg Words: <b>%{y:.1f}</b><extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Content Types (FIXED - Proper pie chart)
        content_data = {
            'Text Messages': len(self.df[~self.df['is_media']]),
            'Media Files': self.df['is_media'].sum(),
            'Questions': self.df['is_question'].sum(),
            'URLs': self.df['has_url'].sum()
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(content_data.keys()),
                values=list(content_data.values()),
                marker=dict(colors=['#4ECDC4', '#FF6B6B', '#96CEB4', '#FFEAA7']),
                textinfo='label+percent+value',
                textfont=dict(size=11),
                hovertemplate='<b>%{label}</b><br>Count: <b>%{value:,}</b><br>Percentage: <b>%{percent}</b><extra></extra>'
            ),
            row=3, col=2
        )
        
        # 7. Hourly Heatmap (FIXED)
        hourly_daily = self.df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        hourly_daily = hourly_daily.reindex(days_order)
        
        fig.add_trace(
            go.Heatmap(
                z=hourly_daily.values,
                x=[f"{h}:00" for h in hourly_daily.columns],
                y=hourly_daily.index,
                colorscale='Viridis',
                showscale=True,
                hovertemplate='<b>%{y}</b> at <b>%{x}</b><br>Messages: <b>%{z:,}</b><extra></extra>',
                colorbar=dict(title=dict(text="Messages", side="right"))
            ),
            row=4, col=1
        )
        
        # 8. Daily Message Volume (FIXED)
        daily_counts = self.df.groupby(self.df['timestamp'].dt.date).size().tail(30)  # Last 30 days
        
        fig.add_trace(
            go.Bar(
                x=[str(d) for d in daily_counts.index],
                y=daily_counts.values,
                name='Daily Volume',
                marker=dict(color='#45B7D1', line=dict(color='rgba(0,0,0,0.3)', width=1)),
                text=[f"{v:,}" for v in daily_counts.values],
                textposition='outside',
                textfont=dict(size=8, color='black'),
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><extra></extra>'
            ),
            row=4, col=2
        )
        
        # 9. Sentiment Distribution (FIXED)
        sentiments = []
        for msg in self.df[~self.df['is_media'] & ~self.df['is_deleted']]['message'].sample(min(1000, len(self.df))):
            try:
                blob = TextBlob(msg)
                sentiment = blob.sentiment.polarity
                if sentiment > 0.1:
                    sentiments.append('Positive')
                elif sentiment < -0.1:
                    sentiments.append('Negative')
                else:
                    sentiments.append('Neutral')
            except:
                sentiments.append('Neutral')
        
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=['Positive 😊', 'Neutral 😐', 'Negative 😔'],
                values=[sentiment_counts.get('Positive', 0), 
                       sentiment_counts.get('Neutral', 0), 
                       sentiment_counts.get('Negative', 0)],
                marker=dict(colors=['#28a745', '#6c757d', '#dc3545']),
                textinfo='label+percent',
                textfont=dict(size=11),
                hovertemplate='<b>%{label}</b><br>Messages: <b>%{value:,}</b><br>Percentage: <b>%{percent}</b><extra></extra>'
            ),
            row=5, col=1
        )
        
        # 10. Time Period Activity (FIXED)
        time_period_data = self.df['time_period'].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=time_period_data.index,
                y=time_period_data.values,
                name='Time Periods',
                marker=dict(color='#FFEAA7', line=dict(color='rgba(0,0,0,0.3)', width=1)),
                text=[f"{v:,}" for v in time_period_data.values],
                textposition='outside',
                textfont=dict(size=10, color='black'),
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><extra></extra>'
            ),
            row=5, col=2
        )
        
        # 11. Top Words (FIXED)
        all_words = ' '.join(self.df[~self.df['is_media']]['message']).lower()
        all_words = re.sub(r'[^\w\s]', ' ', all_words)
        word_list = [word for word in all_words.split() if len(word) > 3 and word.isalpha()]
        word_freq = Counter(word_list).most_common(15)
        
        if word_freq:
            words, counts = zip(*word_freq)
            fig.add_trace(
                go.Bar(
                    x=list(words),
                    y=list(counts),
                    name='Top Words',
                    marker=dict(color='#DDA0DD', line=dict(color='rgba(0,0,0,0.3)', width=1)),
                    text=[f"{v:,}" for v in counts],
                    textposition='outside',
                    textfont=dict(size=9, color='black'),
                    hovertemplate='<b>%{x}</b><br>Count: <b>%{y:,}</b><extra></extra>'
                ),
                row=6, col=1
            )
        
        # 12. Top Emojis (FIXED)
        all_emojis = []
        for msg in self.df['message']:
            all_emojis.extend([c for c in msg if c in emoji.EMOJI_DATA])
        
        if all_emojis:
            emoji_freq = Counter(all_emojis).most_common(10)
            emojis, emoji_counts = zip(*emoji_freq)
            
            fig.add_trace(
                go.Bar(
                    x=list(emojis),
                    y=list(emoji_counts),
                    name='Top Emojis',
                    marker=dict(color='#98D8C8', line=dict(color='rgba(0,0,0,0.3)', width=1)),
                    text=[f"{v:,}" for v in emoji_counts],
                    textposition='outside',
                    textfont=dict(size=12, color='black'),
                    hovertemplate='<b>%{x}</b><br>Count: <b>%{y:,}</b><extra></extra>'
                ),
                row=6, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=2400,
            showlegend=False,
            title={
                'text': "📱 Clean WhatsApp Analytics Dashboard<br><sub>Fixed Colors, Orientations & Data Accuracy</sub>",
                'x': 0.5,
                'font': {'size': 24, 'color': '#2c3e50'}
            },
            font=dict(size=11, family="Arial"),
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa'
        )
        
        # Update axes
        fig.update_xaxes(
            tickangle=-45,
            title_font_size=11,
            tickfont_size=9,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=1
        )
        fig.update_yaxes(
            title_font_size=11,
            tickfont_size=9,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=1
        )
        
        return fig
    
    def create_interactive_network(self):
        """Create clean interactive network"""
        G = nx.Graph()
        
        # Add nodes
        participant_counts = self.df['sender'].value_counts()
        for participant in participant_counts.index:
            G.add_node(participant, size=participant_counts[participant])
        
        # Add edges based on conversation flow
        response_pairs = defaultdict(int)
        for i in range(1, len(self.df)):
            if self.df.iloc[i]['sender'] != self.df.iloc[i-1]['sender']:
                sender = self.df.iloc[i-1]['sender']
                responder = self.df.iloc[i]['sender']
                response_pairs[f"{sender}->{responder}"] += 1
        
        # Add significant connections
        for pair, count in response_pairs.items():
            if count > 10:  # Only significant interactions
                sender, responder = pair.split('->')
                if sender in G.nodes and responder in G.nodes:
                    G.add_edge(sender, responder, weight=count)
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Create network visualization
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='rgba(100,100,100,0.5)'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Node trace
        node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node.split()[0] if ' ' in node else node[:10])
            size = participant_counts[node]
            node_size.append(max(20, min(80, size / 30)))
            node_color.append(size)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial Black'),
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=dict(text="Messages", side="right")),
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>Messages: %{marker.color:,}<extra></extra>',
            showlegend=False
        )
        
        network_fig = go.Figure(data=edge_traces + [node_trace])
        network_fig.update_layout(
            title=dict(text='🤝 Participant Interaction Network', x=0.5, font=dict(size=18)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=500
        )
        
        return network_fig
    
    def generate_clean_report(self, output_dir='clean_analysis'):
        """Generate clean, optimized report"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.parse_messages()
        
        # Create visualizations
        main_fig = self.create_clean_visualizations()
        network_fig = self.create_interactive_network()
        
        # Generate insights
        insights = self._generate_insights()
        
        # Create word cloud
        wordcloud_fig = self._create_wordcloud()
        wordcloud_path = os.path.join(output_dir, 'clean_wordcloud.png')
        wordcloud_fig.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
        plt.close(wordcloud_fig)
        
        # Generate HTML
        html_content = self._generate_clean_html(main_fig, network_fig, insights)
        
        # Save report
        report_path = os.path.join(output_dir, 'clean_whatsapp_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path, insights
    
    def _generate_insights(self):
        """Generate key insights"""
        daily_counts = self.df.groupby(self.df['timestamp'].dt.date).size()
        hourly_counts = self.df['hour'].value_counts()
        participant_counts = self.df['sender'].value_counts()
        
        return {
            'total_messages': len(self.df),
            'total_participants': len(self.participants),
            'date_range': {
                'start': self.df['timestamp'].min(),
                'end': self.df['timestamp'].max(),
                'duration': (self.df['timestamp'].max() - self.df['timestamp'].min()).days
            },
            'busiest_day': {
                'date': daily_counts.idxmax(),
                'count': daily_counts.max()
            },
            'peak_hour': {
                'hour': hourly_counts.idxmax(),
                'count': hourly_counts.max()
            },
            'top_participant': {
                'name': participant_counts.index[0],
                'count': participant_counts.iloc[0]
            },
            'avg_daily': daily_counts.mean()
        }
    
    def _create_wordcloud(self):
        """Create word cloud"""
        text = ' '.join(self.df[~self.df['is_media']]['message'])
        text = re.sub(r'[^\w\s]', ' ', text)
        
        wordcloud = WordCloud(
            width=1000, height=500,
            background_color='white',
            max_words=80,
            colormap='viridis',
            collocations=False
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Most Used Words', fontsize=16, fontweight='bold')
        
        return fig
    
    def _generate_clean_html(self, main_fig, network_fig, insights):
        """Generate clean HTML report"""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>📱 Clean WhatsApp Analytics</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }}
                .header h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; padding: 30px; background: #f8f9fa; }}
                .stat {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .stat-number {{ font-size: 2rem; font-weight: bold; color: #667eea; }}
                .stat-label {{ color: #666; margin-top: 5px; }}
                .section {{ padding: 30px; }}
                .chart {{ background: white; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h2 {{ color: #333; margin-bottom: 20px; }}
                .insights {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                .insight {{ background: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; }}
                .insight-title {{ font-weight: bold; color: #333; }}
                .insight-value {{ color: #667eea; font-size: 1.3rem; font-weight: bold; }}
                .footer {{ background: #333; color: white; text-align: center; padding: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📱 Clean WhatsApp Analytics</h1>
                    <p>Fixed Colors, Orientations & Optimized Visualizations</p>
                    <p>{insights['date_range']['start'].strftime('%B %d, %Y')} - {insights['date_range']['end'].strftime('%B %d, %Y')}</p>
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <div class="stat-number">{insights['total_messages']:,}</div>
                        <div class="stat-label">Total Messages</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{insights['total_participants']}</div>
                        <div class="stat-label">Participants</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{insights['date_range']['duration']}</div>
                        <div class="stat-label">Days</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{insights['avg_daily']:.0f}</div>
                        <div class="stat-label">Avg Daily Messages</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Analytics Dashboard</h2>
                    <div class="chart">
                        <div id="main-dashboard" style="width:100%;height:2400px;"></div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🤝 Network Analysis</h2>
                    <div class="chart">
                        <div id="network-chart" style="width:100%;height:500px;"></div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Key Insights</h2>
                    <div class="insights">
                        <div class="insight">
                            <div class="insight-title">🔥 Busiest Day</div>
                            <div class="insight-value">{insights['busiest_day']['date'].strftime('%B %d, %Y')}</div>
                            <div>{insights['busiest_day']['count']:,} messages</div>
                        </div>
                        <div class="insight">
                            <div class="insight-title">⏰ Peak Hour</div>
                            <div class="insight-value">{insights['peak_hour']['hour']}:00</div>
                            <div>{insights['peak_hour']['count']:,} messages</div>
                        </div>
                        <div class="insight">
                            <div class="insight-title">👑 Most Active</div>
                            <div class="insight-value">{insights['top_participant']['name']}</div>
                            <div>{insights['top_participant']['count']:,} messages</div>
                        </div>
                    </div>
                </div>
                
                <div class="section" style="text-align: center;">
                    <h2>☁️ Word Cloud</h2>
                    <img src="clean_wordcloud.png" alt="Word Cloud" style="max-width: 100%; border-radius: 10px;">
                </div>
                
                <div class="footer">
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
            </div>
            
            <script>
                var mainData = {main_fig.to_json()};
                Plotly.newPlot('main-dashboard', mainData.data, mainData.layout, {{responsive: true}});
                
                var networkData = {network_fig.to_json()};
                Plotly.newPlot('network-chart', networkData.data, networkData.layout, {{responsive: true}});
            </script>
        </body>
        </html>
        """

def main():
    parser = argparse.ArgumentParser(description='Clean WhatsApp Analytics Tool')
    parser.add_argument('file_path', help='Path to WhatsApp chat export file')
    parser.add_argument('--output', '-o', default='clean_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File '{args.file_path}' not found.")
        return
    
    print("🧹 Starting Clean WhatsApp Analytics...")
    print(f"📁 Input file: {args.file_path}")
    print(f"📁 Output directory: {args.output}")
    
    try:
        analyzer = CleanWhatsAppAnalyzer(args.file_path)
        report_path, insights = analyzer.generate_clean_report(args.output)
        
        print(f"\n✅ Clean Analysis Complete!")
        print(f"📊 Processed {insights['total_messages']:,} messages")
        print(f"👥 Found {insights['total_participants']} participants")
        print(f"🔥 Busiest day: {insights['busiest_day']['date'].strftime('%B %d, %Y')} ({insights['busiest_day']['count']:,} messages)")
        print(f"⏰ Peak hour: {insights['peak_hour']['hour']}:00 ({insights['peak_hour']['count']:,} messages)")
        print(f"📝 Clean report: {report_path}")
        print(f"\n🌐 Open {report_path} to view the clean analytics!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
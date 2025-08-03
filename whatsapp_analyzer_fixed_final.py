#!/usr/bin/env python3
"""
Fixed WhatsApp Analytics - Corrected Axis Scaling Issues
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

class FixedFinalWhatsAppAnalyzer:
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
        print(f"📊 Analyzing {len(self.messages)} messages from {len(self.participants)} participants")
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
    
    def create_fixed_visualizations(self):
        """Create visualizations with CORRECT axis scaling and proper data display"""
        
        # First ensure we have data
        if self.df is None or len(self.df) == 0:
            raise ValueError("No data available for visualization")
        
        # Create dashboard with proper scaling
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Messages Sent by Each Person', 'Messages by Hour of Day',
                'Messages by Day of Week', 'Monthly Message Trends',
                'Average Words per Message by Person', 'Content Types Distribution',
                'Time Period Activity', 'Top 10 Most Used Words'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # 1. Messages by Participant - FIXED SCALING
        participant_counts = self.df['sender'].value_counts().head(15)
        print(f"DEBUG: Participant counts: {participant_counts.to_dict()}")
        
        fig.add_trace(
            go.Bar(
                x=list(participant_counts.index),
                y=list(participant_counts.values),
                name='Messages',
                marker=dict(color='#3498db'),
                text=[f"{v:,}" for v in participant_counts.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Hourly Activity - FIXED SCALING  
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
        
        fig.add_trace(
            go.Bar(
                x=hour_labels,
                y=hourly_data.values,
                name='Hourly Messages',
                marker=dict(color='#e74c3c'),
                text=[f"{v:,}" for v in hourly_data.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Daily Activity - FIXED SCALING
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_data = self.df['day_of_week'].value_counts().reindex(days_order, fill_value=0)
        
        fig.add_trace(
            go.Bar(
                x=daily_data.index,
                y=daily_data.values,
                name='Daily Messages',
                marker=dict(color='#2ecc71'),
                text=[f"{v:,}" for v in daily_data.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Monthly Trends - FIXED SCALING
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
                line=dict(color='#f39c12', width=4),
                marker=dict(size=12, color='#f39c12'),
                text=[f"{v:,}" for v in month_values],
                textposition='top center',
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Average Words - FIXED SCALING
        words_by_participant = self.df.groupby('sender')['word_count'].mean().sort_values(ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(
                x=words_by_participant.index,
                y=words_by_participant.values,
                name='Avg Words',
                marker=dict(color='#9b59b6'),
                text=[f"{v:.1f}" for v in words_by_participant.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg Words: <b>%{y:.1f}</b><extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Content Types - PIE CHART
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
                marker=dict(colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']),
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: <b>%{value:,}</b><extra></extra>'
            ),
            row=3, col=2
        )
        
        # 7. Time Period Activity - FIXED SCALING
        time_period_data = self.df['time_period'].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=time_period_data.index,
                y=time_period_data.values,
                name='Time Periods',
                marker=dict(color='#1abc9c'),
                text=[f"{v:,}" for v in time_period_data.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><extra></extra>'
            ),
            row=4, col=1
        )
        
        # 8. Top Words - FIXED SCALING
        all_words = ' '.join(self.df[~self.df['is_media']]['message']).lower()
        all_words = re.sub(r'[^\w\s]', ' ', all_words)
        word_list = [word for word in all_words.split() if len(word) > 3 and word.isalpha()]
        word_freq = Counter(word_list).most_common(10)
        
        if word_freq:
            words, counts = zip(*word_freq)
            fig.add_trace(
                go.Bar(
                    x=list(words),
                    y=list(counts),
                    name='Top Words',
                    marker=dict(color='#34495e'),
                    text=[f"{v:,}" for v in counts],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Count: <b>%{y:,}</b><extra></extra>'
                ),
                row=4, col=2
            )
        
        # CRITICAL FIX: Force proper Y-axis scaling
        fig.update_layout(
            height=1600,
            showlegend=False,
            title={
                'text': "📱 WhatsApp Group Analytics Dashboard<br><sub>Corrected Axis Scaling & Accurate Data Display</sub>",
                'x': 0.5,
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            font=dict(size=11),
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa'
        )
        
        # FORCE CORRECT Y-AXIS RANGES
        max_participant = participant_counts.max()
        fig.update_yaxes(range=[0, max_participant * 1.1], row=1, col=1)
        fig.update_yaxes(title_text="Number of Messages", row=1, col=1)
        fig.update_xaxes(title_text="Participants", row=1, col=1)
        
        max_hourly = hourly_data.max()
        fig.update_yaxes(range=[0, max_hourly * 1.1], row=1, col=2)
        fig.update_yaxes(title_text="Messages Sent", row=1, col=2)
        fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
        
        max_daily = daily_data.max()
        fig.update_yaxes(range=[0, max_daily * 1.1], row=2, col=1)
        fig.update_yaxes(title_text="Total Messages", row=2, col=1)
        fig.update_xaxes(title_text="Day of Week", row=2, col=1)
        
        max_monthly = max(month_values)
        fig.update_yaxes(range=[0, max_monthly * 1.1], row=2, col=2)
        fig.update_yaxes(title_text="Message Count", row=2, col=2)
        fig.update_xaxes(title_text="Month", row=2, col=2)
        
        max_words = words_by_participant.max()
        fig.update_yaxes(range=[0, max_words * 1.1], row=3, col=1)
        fig.update_yaxes(title_text="Average Words", row=3, col=1)
        fig.update_xaxes(title_text="Participants", row=3, col=1)
        
        max_time_period = time_period_data.max()
        fig.update_yaxes(range=[0, max_time_period * 1.1], row=4, col=1)
        fig.update_yaxes(title_text="Messages", row=4, col=1)
        fig.update_xaxes(title_text="Time Period", row=4, col=1)
        
        if word_freq:
            max_word_count = max(counts)
            fig.update_yaxes(range=[0, max_word_count * 1.1], row=4, col=2)
            fig.update_yaxes(title_text="Word Count", row=4, col=2)
            fig.update_xaxes(title_text="Words", row=4, col=2)
        
        # Update all X-axis formatting
        fig.update_xaxes(tickangle=-45)
        
        return fig
    
    def generate_comprehensive_statistics(self):
        """Generate comprehensive statistics with detailed breakdowns"""
        daily_counts = self.df.groupby(self.df['timestamp'].dt.date).size()
        hourly_counts = self.df['hour'].value_counts()
        participant_counts = self.df['sender'].value_counts()
        
        # Participant detailed stats
        participant_stats = []
        for participant in participant_counts.index[:10]:
            participant_data = self.df[self.df['sender'] == participant]
            stats = {
                'name': participant,
                'messages': len(participant_data),
                'words': participant_data['word_count'].sum(),
                'avg_words': participant_data['word_count'].mean(),
                'characters': participant_data['char_count'].sum(),
                'media_files': participant_data['is_media'].sum(),
                'questions': participant_data['is_question'].sum(),
                'first_message': participant_data['timestamp'].min().strftime('%B %d, %Y'),
                'last_message': participant_data['timestamp'].max().strftime('%B %d, %Y')
            }
            participant_stats.append(stats)
        
        # Time-based analysis
        day_stats = []
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days_order:
            day_data = self.df[self.df['day_of_week'] == day]
            day_stats.append({
                'day': day,
                'messages': len(day_data),
                'percentage': (len(day_data) / len(self.df)) * 100,
                'avg_words': day_data['word_count'].mean() if len(day_data) > 0 else 0
            })
        
        # Hour breakdown
        hour_stats = []
        for hour in range(24):
            hour_data = self.df[self.df['hour'] == hour]
            hour_label = f"{hour:02d}:00"
            if hour == 0:
                hour_label = "12:00 AM"
            elif hour < 12:
                hour_label = f"{hour}:00 AM"
            elif hour == 12:
                hour_label = "12:00 PM"
            else:
                hour_label = f"{hour-12}:00 PM"
                
            hour_stats.append({
                'hour': hour_label,
                'messages': len(hour_data),
                'percentage': (len(hour_data) / len(self.df)) * 100 if len(hour_data) > 0 else 0
            })
        
        return {
            'total_messages': len(self.df),
            'total_participants': len(self.participants),
            'date_range': {
                'start': self.df['timestamp'].min().strftime('%B %d, %Y'),
                'end': self.df['timestamp'].max().strftime('%B %d, %Y'),
                'duration': (self.df['timestamp'].max() - self.df['timestamp'].min()).days
            },
            'busiest_date': {
                'date': daily_counts.idxmax().strftime('%B %d, %Y'),
                'count': daily_counts.max()
            },
            'peak_hour': {
                'hour': hourly_counts.idxmax(),
                'count': hourly_counts.max(),
                'formatted': f"{hourly_counts.idxmax()}:00"
            },
            'top_participant': {
                'name': participant_counts.index[0],
                'count': participant_counts.iloc[0]
            },
            'avg_daily': daily_counts.mean(),
            'participant_details': participant_stats,
            'day_breakdown': day_stats,
            'hour_breakdown': hour_stats,
            'content_stats': {
                'text_messages': len(self.df[~self.df['is_media']]),
                'media_files': self.df['is_media'].sum(),
                'questions': self.df['is_question'].sum(),
                'urls': self.df['has_url'].sum(),
                'total_words': self.df['word_count'].sum(),
                'total_characters': self.df['char_count'].sum(),
                'avg_message_length': self.df['word_count'].mean()
            }
        }
    
    def generate_corrected_report(self, output_dir='corrected_analysis'):
        """Generate corrected report with proper scaling"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.parse_messages()
        
        # Create corrected visualizations
        main_fig = self.create_fixed_visualizations()
        stats = self.generate_comprehensive_statistics()
        
        # Create word cloud
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
        
        wordcloud_path = os.path.join(output_dir, 'wordcloud_corrected.png')
        fig.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Generate HTML with explanations
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>WhatsApp Analytics - Corrected Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: 'Arial', sans-serif; background: #f5f5f5; }}
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
                .explanation {{ background: #e8f4fd; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #667eea; }}
                .insights {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                .insight {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; }}
                .insight-title {{ font-weight: bold; color: #333; }}
                .insight-value {{ color: #28a745; font-size: 1.3rem; font-weight: bold; }}
                .footer {{ background: #333; color: white; text-align: center; padding: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📱 WhatsApp Analytics Report</h1>
                    <p>Corrected Data Visualization & Accurate Scaling</p>
                    <p>{stats['date_range']['start']} - {stats['date_range']['end']}</p>
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <div class="stat-number">{stats['total_messages']:,}</div>
                        <div class="stat-label">Total Messages</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['total_participants']}</div>
                        <div class="stat-label">Participants</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['date_range']['duration']}</div>
                        <div class="stat-label">Days</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['avg_daily']:.0f}</div>
                        <div class="stat-label">Avg Daily Messages</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Corrected Analytics Dashboard</h2>
                    <div class="explanation">
                        <strong>✅ Fixed Issues:</strong><br>
                        • Y-axis scaling now matches actual data values<br>
                        • Bar heights correctly represent the numbers shown<br>
                        • No more contradictory information between axes and data<br>
                        • All charts properly scaled to their data ranges
                    </div>
                    <div class="chart">
                        <div id="main-dashboard" style="width:100%;height:1600px;"></div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Key Insights</h2>
                    <div class="insights">
                        <div class="insight">
                            <div class="insight-title">🔥 Busiest Day</div>
                            <div class="insight-value">{stats['busiest_date']['date']}</div>
                            <div>{stats['busiest_date']['count']:,} messages</div>
                        </div>
                        <div class="insight">
                            <div class="insight-title">⏰ Peak Hour</div>
                            <div class="insight-value">{stats['peak_hour']['formatted']}</div>
                            <div>{stats['peak_hour']['count']:,} messages</div>
                        </div>
                        <div class="insight">
                            <div class="insight-title">👑 Most Active</div>
                            <div class="insight-value">{stats['top_participant']['name']}</div>
                            <div>{stats['top_participant']['count']:,} messages</div>
                        </div>
                    </div>
                </div>
                
                <div class="section" style="text-align: center;">
                    <h2>☁️ Word Cloud</h2>
                    <img src="wordcloud_corrected.png" alt="Word Cloud" style="max-width: 100%; border-radius: 10px;">
                </div>
                
                <div class="footer">
                    <p>📊 Corrected WhatsApp Analytics Report</p>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    <p style="margin-top: 10px; font-size: 0.9rem;">All axis scaling issues have been resolved</p>
                </div>
            </div>
            
            <script>
                var mainData = {main_fig.to_json()};
                Plotly.newPlot('main-dashboard', mainData.data, mainData.layout, {{responsive: true}});
            </script>
        </body>
        </html>
        """
        
        report_path = os.path.join(output_dir, 'corrected_analytics_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path, stats

def main():
    parser = argparse.ArgumentParser(description='Fixed WhatsApp Analytics Tool')
    parser.add_argument('file_path', help='Path to WhatsApp chat export file')
    parser.add_argument('--output', '-o', default='corrected_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File '{args.file_path}' not found.")
        return
    
    print("🔧 Starting CORRECTED WhatsApp Analytics...")
    print(f"📁 Input file: {args.file_path}")
    print(f"📁 Output directory: {args.output}")
    
    try:
        analyzer = FixedFinalWhatsAppAnalyzer(args.file_path)
        report_path, stats = analyzer.generate_corrected_report(args.output)
        
        print(f"\n✅ CORRECTED Analysis Complete!")
        print(f"📊 Analyzed {stats['total_messages']:,} messages")
        print(f"👥 From {stats['total_participants']} participants")
        print(f"🔥 Busiest day: {stats['busiest_date']['date']} ({stats['busiest_date']['count']:,} messages)")
        print(f"⏰ Peak hour: {stats['peak_hour']['formatted']} ({stats['peak_hour']['count']:,} messages)")
        print(f"📝 CORRECTED report: {report_path}")
        print(f"\n🌐 ✅ All axis scaling issues have been FIXED!")
        print(f"Open {report_path} to view the corrected analytics!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
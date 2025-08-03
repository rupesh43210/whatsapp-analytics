#!/usr/bin/env python3
"""
Enhanced WhatsApp Chat Analytics Tool
Creates comprehensive, polished analytics and visualizations from WhatsApp chat exports
With advanced insights, relationship analysis, and beautiful modern design
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

class EnhancedWhatsAppAnalyzer:
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
            # Format: DD.MM.YY, HH:MM - Name: Message
            r'(\d{1,2}\.\d{1,2}\.\d{2,4}),?\s+(\d{1,2}:\d{2})\s*[-–]\s*([^:]+?):\s*(.*)',
            # Format: YYYY-MM-DD HH:MM:SS - Name: Message
            r'(\d{4}-\d{1,2}-\d{1,2})\s+(\d{1,2}:\d{2}:\d{2})\s*[-–]\s*([^:]+?):\s*(.*)',
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
        
        # Calculate conversation threads
        self._identify_conversation_patterns()
        
    def _identify_conversation_patterns(self):
        """Identify conversation initiators, responders, and interaction patterns"""
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Identify conversation starters (messages after long gaps)
        conversation_gaps = []
        initiators = []
        responders = []
        
        for i in range(1, len(self.df)):
            time_diff = (self.df.iloc[i]['timestamp'] - self.df.iloc[i-1]['timestamp']).total_seconds() / 3600
            
            if time_diff > 2:  # 2+ hour gap = new conversation
                conversation_gaps.append(True)
                initiators.append(self.df.iloc[i]['sender'])
                if i > 0:
                    responders.append(self.df.iloc[i-1]['sender'])
            else:
                conversation_gaps.append(False)
        
        # Add conversation gap indicator
        self.df['is_conversation_starter'] = [False] + conversation_gaps
        
        # Store conversation patterns
        self.conversation_initiators = Counter(initiators)
        self.conversation_enders = Counter(responders)
        
        # Calculate response patterns
        self._calculate_response_patterns()
        
    def _calculate_response_patterns(self):
        """Calculate detailed response patterns and relationships"""
        self.response_matrix = defaultdict(lambda: defaultdict(int))
        self.avg_response_times = defaultdict(list)
        
        for i in range(1, len(self.df)):
            current_sender = self.df.iloc[i]['sender']
            prev_sender = self.df.iloc[i-1]['sender']
            
            if current_sender != prev_sender:
                time_diff = (self.df.iloc[i]['timestamp'] - self.df.iloc[i-1]['timestamp']).total_seconds() / 60
                
                if time_diff < 60:  # Response within an hour
                    self.response_matrix[prev_sender][current_sender] += 1
                    self.avg_response_times[f"{prev_sender} -> {current_sender}"].append(time_diff)
    
    def generate_enhanced_analytics(self):
        """Generate comprehensive analytics with advanced insights"""
        if self.df is None or self.df.empty:
            raise ValueError("No data to analyze. Please parse messages first.")
        
        analytics = {}
        
        # Basic enhanced stats
        analytics['total_messages'] = len(self.df)
        analytics['total_participants'] = len(self.participants)
        analytics['date_range'] = {
            'start': self.df['timestamp'].min(),
            'end': self.df['timestamp'].max(),
            'duration_days': (self.df['timestamp'].max() - self.df['timestamp'].min()).days,
            'start_formatted': self.df['timestamp'].min().strftime('%B %d, %Y at %I:%M %p'),
            'end_formatted': self.df['timestamp'].max().strftime('%B %d, %Y at %I:%M %p')
        }
        
        # Enhanced participant statistics
        participant_stats = {}
        for participant in self.participants:
            p_data = self.df[self.df['sender'] == participant]
            participant_stats[participant] = {
                'total_messages': len(p_data),
                'total_words': p_data['word_count'].sum(),
                'avg_words_per_message': p_data['word_count'].mean(),
                'total_characters': p_data['char_count'].sum(),
                'avg_chars_per_message': p_data['char_count'].mean(),
                'total_emojis': p_data['emoji_count'].sum(),
                'media_messages': p_data['is_media'].sum(),
                'questions_asked': p_data['is_question'].sum(),
                'urls_shared': p_data['has_url'].sum(),
                'conversation_starters': p_data['is_conversation_starter'].sum(),
                'first_message': p_data['timestamp'].min(),
                'last_message': p_data['timestamp'].max(),
                'most_active_hour': p_data['hour'].mode().iloc[0] if not p_data['hour'].mode().empty else 0,
                'most_active_day': p_data['day_of_week'].mode().iloc[0] if not p_data['day_of_week'].mode().empty else 'Unknown'
            }
        
        analytics['participant_stats'] = participant_stats
        
        # Time-based analytics with better granularity
        analytics['hourly_activity'] = self.df['hour'].value_counts().to_dict()
        analytics['daily_activity'] = self.df['day_of_week'].value_counts().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]).fillna(0).to_dict()
        
        # Monthly and weekly trends
        monthly_data = self.df.groupby([self.df['timestamp'].dt.year, self.df['timestamp'].dt.month]).size()
        analytics['monthly_trends'] = {f"{month_name} {year}": count 
                                     for (year, month), count in monthly_data.items() 
                                     for month_name in [datetime(year, month, 1).strftime('%B')]}
        
        # Time period analysis
        analytics['time_period_activity'] = self.df['time_period'].value_counts().to_dict()
        
        # Enhanced word and emoji analysis
        all_words = ' '.join(self.df[~self.df['is_media']]['message']).lower()
        # Clean text better
        all_words = re.sub(r'[^\w\s]', ' ', all_words)
        word_list = [word for word in all_words.split() if len(word) > 2 and word.isalpha()]
        word_freq = Counter(word_list)
        analytics['top_words'] = dict(word_freq.most_common(30))
        
        # Emoji analysis
        all_emojis = []
        for msg in self.df['message']:
            all_emojis.extend([c for c in msg if c in emoji.EMOJI_DATA])
        emoji_freq = Counter(all_emojis)
        analytics['top_emojis'] = dict(emoji_freq.most_common(15))
        
        # Media and content analysis
        analytics['content_analysis'] = {
            'total_media': self.df['is_media'].sum(),
            'total_deleted': self.df['is_deleted'].sum(),
            'total_questions': self.df['is_question'].sum(),
            'total_urls': self.df['has_url'].sum(),
            'avg_message_length': self.df['char_count'].mean(),
            'longest_message': self.df['char_count'].max(),
            'most_words_in_message': self.df['word_count'].max()
        }
        
        # Advanced sentiment analysis
        analytics['sentiment_analysis'] = self._advanced_sentiment_analysis()
        
        # Conversation patterns
        analytics['conversation_patterns'] = {
            'top_initiators': dict(self.conversation_initiators.most_common(10)),
            'top_conversation_enders': dict(self.conversation_enders.most_common(10)),
            'response_matrix': dict(self.response_matrix),
            'avg_response_times': {k: np.mean(v) for k, v in self.avg_response_times.items() if v}
        }
        
        # Peak activity analysis
        peak_hour = max(analytics['hourly_activity'], key=analytics['hourly_activity'].get)
        peak_day = max(analytics['daily_activity'], key=analytics['daily_activity'].get)
        
        analytics['peak_activity'] = {
            'peak_hour': peak_hour,
            'peak_hour_messages': analytics['hourly_activity'][peak_hour],
            'peak_day': peak_day,
            'peak_day_messages': analytics['daily_activity'][peak_day],
            'busiest_participant': max(participant_stats, key=lambda x: participant_stats[x]['total_messages']),
            'most_verbose_participant': max(participant_stats, key=lambda x: participant_stats[x]['avg_words_per_message']),
            'emoji_king': max(participant_stats, key=lambda x: participant_stats[x]['total_emojis']),
            'question_master': max(participant_stats, key=lambda x: participant_stats[x]['questions_asked'])
        }
        
        return analytics
    
    def _advanced_sentiment_analysis(self):
        """Perform advanced sentiment analysis with trends"""
        sentiments = []
        sentiment_by_participant = defaultdict(list)
        sentiment_over_time = []
        
        for _, row in self.df.iterrows():
            if not row['is_media'] and not row['is_deleted']:
                try:
                    blob = TextBlob(row['message'])
                    sentiment = blob.sentiment.polarity
                    sentiments.append(sentiment)
                    sentiment_by_participant[row['sender']].append(sentiment)
                    sentiment_over_time.append({
                        'date': row['timestamp'],
                        'sentiment': sentiment,
                        'sender': row['sender']
                    })
                except:
                    sentiments.append(0)
        
        # Calculate rolling sentiment trends
        sentiment_df = pd.DataFrame(sentiment_over_time)
        if not sentiment_df.empty:
            sentiment_df = sentiment_df.sort_values('date')
            sentiment_df['rolling_sentiment'] = sentiment_df['sentiment'].rolling(window=50, center=True).mean()
        
        return {
            'overall_sentiment': np.mean(sentiments) if sentiments else 0,
            'sentiment_by_participant': {k: np.mean(v) for k, v in sentiment_by_participant.items()},
            'sentiment_distribution': {
                'positive': len([s for s in sentiments if s > 0.1]),
                'neutral': len([s for s in sentiments if -0.1 <= s <= 0.1]),
                'negative': len([s for s in sentiments if s < -0.1])
            },
            'sentiment_trends': sentiment_df.to_dict('records') if not sentiment_df.empty else []
        }
    
    def create_enhanced_visualizations(self, analytics):
        """Create comprehensive, crystal-clear visualizations with proper labeling"""
        # Create main dashboard with proper spacing
        fig = make_subplots(
            rows=8, cols=2,
            subplot_titles=[
                '📊 Total Messages Sent by Each Participant', '⏰ Messages Sent by Hour of Day (24-hour format)',
                '📅 Messages Sent by Day of Week', '📈 Monthly Message Volume Trends',
                '💭 Average Words per Message Distribution', '😊 Message Sentiment Distribution',
                '🎯 Types of Content Shared', '🔥 Activity by Time of Day',
                '💬 Who Starts Conversations Most', '🤝 Average Response Time (minutes)',
                '🌟 Most Frequently Used Words', '😀 Most Popular Emojis',
                '📱 Messages by Time Period', '🎪 Participant Engagement Score',
                '📊 Daily Sentiment Trends', '🎨 Average Characters per Message'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "pie"}],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.12
        )
        
        # 1. Messages by Participant (sorted with better formatting)
        participant_counts = pd.Series({k: v['total_messages'] for k, v in analytics['participant_stats'].items()}).sort_values(ascending=False)
        colors = px.colors.qualitative.Set3[:len(participant_counts)]
        fig.add_trace(
            go.Bar(x=participant_counts.index, y=participant_counts.values, 
                   name='Total Messages', marker_color=colors,
                   text=[f"{v:,}" for v in participant_counts.values], 
                   textposition='outside',
                   hovertemplate='<b>%{x}</b><br>Messages: %{y:,}<br><extra></extra>'),
            row=1, col=1
        )
        
        # 2. Hourly Activity with clear time labels
        hourly_data = pd.Series(analytics['hourly_activity']).sort_index()
        time_labels = []
        for h in hourly_data.index:
            if h == 0:
                time_labels.append('12:00 AM (Midnight)')
            elif h < 12:
                time_labels.append(f'{h}:00 AM')
            elif h == 12:
                time_labels.append('12:00 PM (Noon)')
            else:
                time_labels.append(f'{h-12}:00 PM')
        
        fig.add_trace(
            go.Bar(x=time_labels, y=hourly_data.values,
                   name='Messages per Hour', marker_color='lightblue',
                   text=[f"{v:,}" for v in hourly_data.values], 
                   textposition='outside',
                   hovertemplate='<b>%{x}</b><br>Messages: %{y:,}<br><extra></extra>'),
            row=1, col=2
        )
        
        # 3. Weekly Activity Pattern
        daily_data = pd.Series(analytics['daily_activity'])
        fig.add_trace(
            go.Bar(x=daily_data.index, y=daily_data.values,
                   name='Messages per Day', marker_color='lightgreen',
                   text=[f"{v:,}" for v in daily_data.values], 
                   textposition='outside',
                   hovertemplate='<b>%{x}</b><br>Messages: %{y:,}<br><extra></extra>'),
            row=2, col=1
        )
        
        # 4. Monthly Trends with clear labels
        monthly_data = pd.Series(analytics['monthly_trends'])
        fig.add_trace(
            go.Scatter(x=list(monthly_data.index), y=monthly_data.values,
                      mode='lines+markers', name='Monthly Message Volume',
                      line=dict(color='orange', width=3),
                      marker=dict(size=10),
                      text=[f"{v:,} messages" for v in monthly_data.values],
                      textposition='top center',
                      hovertemplate='<b>%{x}</b><br>Messages: %{y:,}<br><extra></extra>'),
            row=2, col=2
        )
        
        # 5. Message Length Distribution
        word_counts = [stats['avg_words_per_message'] for stats in analytics['participant_stats'].values()]
        fig.add_trace(
            go.Histogram(x=word_counts, nbinsx=20, name='Message Length',
                        marker_color='purple', opacity=0.7),
            row=3, col=1
        )
        
        # 6. Sentiment Distribution
        sentiment_dist = analytics['sentiment_analysis']['sentiment_distribution']
        fig.add_trace(
            go.Pie(labels=['Positive 😊', 'Neutral 😐', 'Negative 😔'],
                   values=[sentiment_dist['positive'], sentiment_dist['neutral'], sentiment_dist['negative']],
                   marker_colors=['green', 'gray', 'red']),
            row=3, col=2
        )
        
        # 7. Content Type Breakdown
        content_analysis = analytics['content_analysis']
        content_types = ['Text Messages', 'Media', 'Deleted', 'Questions', 'URLs']
        content_values = [
            analytics['total_messages'] - content_analysis['total_media'],
            content_analysis['total_media'],
            content_analysis['total_deleted'],
            content_analysis['total_questions'],
            content_analysis['total_urls']
        ]
        fig.add_trace(
            go.Pie(labels=content_types, values=content_values,
                   marker_colors=['lightblue', 'orange', 'red', 'green', 'purple']),
            row=4, col=1
        )
        
        # 8. Peak Activity Times
        time_period_data = pd.Series(analytics['time_period_activity'])
        fig.add_trace(
            go.Bar(x=time_period_data.index, y=time_period_data.values,
                   name='Time Periods', marker_color='gold',
                   text=time_period_data.values, textposition='outside'),
            row=4, col=2
        )
        
        # 9. Conversation Initiators
        initiators = pd.Series(analytics['conversation_patterns']['top_initiators']).head(10)
        fig.add_trace(
            go.Bar(x=initiators.index, y=initiators.values,
                   name='Initiators', marker_color='lightcoral',
                   text=initiators.values, textposition='outside'),
            row=5, col=1
        )
        
        # 10. Response Network (simplified)
        response_data = analytics['conversation_patterns']['avg_response_times']
        if response_data:
            resp_pairs = list(response_data.keys())[:10]
            resp_times = [response_data[pair] for pair in resp_pairs]
            fig.add_trace(
                go.Scatter(x=resp_pairs, y=resp_times, mode='markers',
                          marker=dict(size=10, color='navy'),
                          name='Response Times'),
                row=5, col=2
            )
        
        # 11. Top Words
        top_words = pd.Series(analytics['top_words']).head(15)
        fig.add_trace(
            go.Bar(x=top_words.index, y=top_words.values,
                   name='Top Words', marker_color='teal',
                   text=top_words.values, textposition='outside'),
            row=6, col=1
        )
        
        # 12. Top Emojis
        top_emojis = pd.Series(analytics['top_emojis']).head(10)
        fig.add_trace(
            go.Bar(x=list(top_emojis.index), y=top_emojis.values,
                   name='Top Emojis', marker_color='pink',
                   text=top_emojis.values, textposition='outside'),
            row=6, col=2
        )
        
        # 13. Participant Engagement Score
        engagement_scores = {}
        for participant, stats in analytics['participant_stats'].items():
            score = (stats['total_messages'] * 0.3 + 
                    stats['conversation_starters'] * 0.4 + 
                    stats['questions_asked'] * 0.2 + 
                    stats['total_emojis'] * 0.1)
            engagement_scores[participant] = score
        
        engagement_data = pd.Series(engagement_scores).sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=engagement_data.index, y=engagement_data.values,
                   name='Engagement Score', marker_color='indigo',
                   text=[f"{v:.1f}" for v in engagement_data.values], textposition='outside'),
            row=7, col=1
        )
        
        # 14. Message Characteristics
        chars_per_participant = {k: v['avg_chars_per_message'] for k, v in analytics['participant_stats'].items()}
        char_data = pd.Series(chars_per_participant).sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=char_data.index, y=char_data.values,
                   name='Avg Characters', marker_color='brown',
                   text=[f"{v:.0f}" for v in char_data.values], textposition='outside'),
            row=7, col=2
        )
        
        # 15. Sentiment Trends Over Time (if data available)
        if analytics['sentiment_analysis']['sentiment_trends']:
            sentiment_df = pd.DataFrame(analytics['sentiment_analysis']['sentiment_trends'])
            # Group by date for better visualization
            daily_sentiment = sentiment_df.groupby(sentiment_df['date'].dt.date)['sentiment'].mean()
            fig.add_trace(
                go.Scatter(x=daily_sentiment.index, y=daily_sentiment.values,
                          mode='lines+markers', name='Daily Sentiment',
                          line=dict(color='green', width=2)),
                row=8, col=1
            )
        
        # 16. Advanced Metrics
        advanced_metrics = ['Media Share', 'Question Rate', 'URL Share', 'Emoji Rate']
        media_rates = [stats['media_messages']/stats['total_messages'] * 100 for stats in analytics['participant_stats'].values()]
        avg_media_rate = np.mean(media_rates)
        
        question_rates = [stats['questions_asked']/stats['total_messages'] * 100 for stats in analytics['participant_stats'].values()]
        avg_question_rate = np.mean(question_rates)
        
        url_rates = [stats['urls_shared']/stats['total_messages'] * 100 for stats in analytics['participant_stats'].values()]
        avg_url_rate = np.mean(url_rates)
        
        emoji_rates = [stats['total_emojis']/stats['total_messages'] * 100 for stats in analytics['participant_stats'].values()]
        avg_emoji_rate = np.mean(emoji_rates)
        
        fig.add_trace(
            go.Bar(x=advanced_metrics, 
                   y=[avg_media_rate, avg_question_rate, avg_url_rate, avg_emoji_rate],
                   name='Usage Rates (%)', marker_color='violet',
                   text=[f"{v:.1f}%" for v in [avg_media_rate, avg_question_rate, avg_url_rate, avg_emoji_rate]], 
                   textposition='outside'),
            row=8, col=2
        )
        
        # Update layout with better formatting
        fig.update_layout(
            height=3200,
            showlegend=False,
            title={
                'text': "📱 Enhanced WhatsApp Chat Analytics Dashboard",
                'x': 0.5,
                'font': {'size': 28, 'color': '#2E86AB'}
            },
            font=dict(size=11),
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa'
        )
        
        # Update all axes with better formatting
        fig.update_xaxes(
            tickangle=-45, 
            title_font_size=12,
            tickfont_size=10,
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        )
        fig.update_yaxes(
            title_font_size=12,
            tickfont_size=10,
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            tickformat=',d'  # Format numbers with commas
        )
        
        return fig
    
    def create_interactive_network(self, analytics):
        """Create a highly interactive network graph with better positioning and hover details"""
        G = nx.Graph()
        
        # Add nodes with detailed information
        for participant in analytics['participant_stats']:
            stats = analytics['participant_stats'][participant]
            G.add_node(participant, 
                      size=stats['total_messages'],
                      engagement=stats['conversation_starters'],
                      questions=stats['questions_asked'],
                      emojis=stats['total_emojis'])
        
        # Add edges with interaction strength
        response_matrix = analytics['conversation_patterns']['response_matrix']
        edge_info = []
        
        for sender, responses in response_matrix.items():
            for responder, count in responses.items():
                if count > 3:  # Lower threshold for more connections
                    G.add_edge(sender, responder, weight=count)
                    edge_info.append({
                        'from': sender,
                        'to': responder,
                        'interactions': count
                    })
        
        # Use force-directed layout with better spacing
        pos = nx.spring_layout(G, k=5, iterations=100, seed=42)
        
        # Create edge traces with varying thickness
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=max(1, weight/3), color='rgba(50,50,50,0.5)'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Create interactive node trace
        node_x = []
        node_y = []
        node_info = []
        node_sizes = []
        node_colors = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            stats = analytics['participant_stats'][node]
            
            # Create detailed hover information
            hover_text = f"""
            <b>{node}</b><br>
            📱 Messages: {stats['total_messages']:,}<br>
            💬 Conversations Started: {stats['conversation_starters']}<br>
            ❓ Questions Asked: {stats['questions_asked']}<br>
            😀 Emojis Used: {stats['total_emojis']}<br>
            📝 Avg Words/Message: {stats['avg_words_per_message']:.1f}<br>
            🎯 Most Active: {stats['most_active_day']} at {stats['most_active_hour']}:00
            """
            node_info.append(hover_text)
            
            # Size based on message count (with better scaling)
            node_sizes.append(max(20, min(60, stats['total_messages'] / 50)))
            
            # Color based on engagement score
            engagement_score = (stats['total_messages'] * 0.4 + 
                              stats['conversation_starters'] * 0.6)
            node_colors.append(engagement_score)
            
            # Display first name or short name
            display_name = node.split()[0] if ' ' in node else node[:8]
            node_text.append(display_name)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=node_info,
            text=node_text,
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial Black'),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(text="Engagement Score", side="right"),
                    tickmode="linear",
                    thickness=15
                ),
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Create the figure with enhanced interactivity
        network_fig = go.Figure(data=edge_traces + [node_trace])
        
        network_fig.update_layout(
            title={
                'text': '🤝 Interactive Participant Network<br><sub>Node size = Messages sent | Color = Engagement score | Hover for details</sub>',
                'font': {'size': 18},
                'x': 0.5
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=80),
            annotations=[
                dict(
                    text="💡 Tip: Hover over nodes for detailed stats | Larger nodes = More messages | Lines = Interactions",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.1,
                    font=dict(size=12, color='gray')
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(240,240,240,0.1)',
            height=600
        )
        
        return network_fig
    
    def generate_enhanced_report(self, output_dir='enhanced_output'):
        """Generate comprehensive enhanced HTML report"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Parse messages and generate analytics
        self.parse_messages()
        analytics = self.generate_enhanced_analytics()
        
        # Create visualizations
        main_fig = self.create_enhanced_visualizations(analytics)
        network_fig = self.create_interactive_network(analytics)
        
        # Create enhanced word cloud
        wordcloud_fig = self.create_enhanced_wordcloud()
        
        # Save visualizations
        wordcloud_path = os.path.join(output_dir, 'enhanced_wordcloud.png')
        wordcloud_fig.savefig(wordcloud_path, dpi=300, bbox_inches='tight', 
                             facecolor='white', edgecolor='none')
        plt.close(wordcloud_fig)
        
        # Generate comprehensive HTML report
        html_content = self._generate_enhanced_html(analytics, main_fig, network_fig, output_dir)
        
        # Save HTML report
        report_path = os.path.join(output_dir, 'enhanced_whatsapp_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save enhanced analytics as JSON
        self._save_enhanced_analytics(analytics, output_dir)
        
        return report_path, analytics
    
    def create_enhanced_wordcloud(self):
        """Create beautiful enhanced word cloud"""
        # Get text data excluding media messages
        text_messages = self.df[~self.df['is_media'] & ~self.df['is_deleted']]['message']
        all_text = ' '.join(text_messages)
        
        # Clean text
        all_text = re.sub(r'http[s]?://\S+', '', all_text)  # Remove URLs
        all_text = re.sub(r'@\w+', '', all_text)  # Remove mentions
        all_text = re.sub(r'[^\w\s]', ' ', all_text)  # Remove punctuation
        
        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            max_words=150,
            colormap='Set3',
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=80,
            collocations=False
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('☁️ Enhanced Word Cloud - Most Used Words', 
                     fontsize=20, fontweight='bold', pad=20)
        
        return fig
    
    def _generate_enhanced_html(self, analytics, main_fig, network_fig, output_dir):
        """Generate comprehensive enhanced HTML report"""
        
        # Get key insights
        peak_activity = analytics['peak_activity']
        date_range = analytics['date_range']
        sentiment = analytics['sentiment_analysis']
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>📱 Enhanced WhatsApp Analytics Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/js/all.min.js"></script>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                
                body {{ 
                    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; 
                    line-height: 1.6; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                
                .container {{ 
                    max-width: 1400px; 
                    margin: 0 auto; 
                    background: white; 
                    box-shadow: 0 20px 60px rgba(0,0,0,0.15);
                    border-radius: 15px;
                    overflow: hidden;
                    margin-top: 20px;
                    margin-bottom: 20px;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #25D366 0%, #128C7E 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                }}
                
                .header::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="rgba(255,255,255,0.1)"/></svg>') repeat;
                    animation: float 20s infinite linear;
                }}
                
                @keyframes float {{
                    0% {{ transform: translate(0, 0) rotate(0deg); }}
                    100% {{ transform: translate(-100px, -100px) rotate(360deg); }}
                }}
                
                .header h1 {{
                    font-size: 3rem;
                    margin-bottom: 15px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                    position: relative;
                    z-index: 1;
                }}
                
                .header .subtitle {{
                    font-size: 1.3rem;
                    opacity: 0.9;
                    position: relative;
                    z-index: 1;
                }}
                
                .overview-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 25px;
                    padding: 40px;
                    background: #f8f9fa;
                }}
                
                .stat-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    border-top: 4px solid;
                }}
                
                .stat-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
                }}
                
                .stat-card.messages {{ border-top-color: #25D366; }}
                .stat-card.participants {{ border-top-color: #128C7E; }}
                .stat-card.days {{ border-top-color: #34495e; }}
                .stat-card.media {{ border-top-color: #e74c3c; }}
                .stat-card.sentiment {{ border-top-color: #f39c12; }}
                .stat-card.peak {{ border-top-color: #9b59b6; }}
                
                .stat-number {{
                    font-size: 2.5rem;
                    font-weight: bold;
                    margin-bottom: 8px;
                }}
                
                .stat-label {{
                    color: #666;
                    font-size: 0.95rem;
                    font-weight: 500;
                }}
                
                .section {{
                    padding: 40px;
                    border-bottom: 1px solid #eee;
                }}
                
                .section h2 {{
                    color: #2c3e50;
                    margin-bottom: 25px;
                    font-size: 2rem;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                
                .section h2 i {{
                    color: #25D366;
                }}
                
                .participants-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 25px 0;
                }}
                
                .participant-card {{
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    padding: 20px;
                    border-radius: 12px;
                    border-left: 5px solid #25D366;
                    transition: all 0.3s ease;
                }}
                
                .participant-card:hover {{
                    transform: translateX(5px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                
                .participant-name {{
                    font-weight: bold;
                    font-size: 1.1rem;
                    color: #2c3e50;
                    margin-bottom: 10px;
                }}
                
                .participant-stats {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 8px;
                    font-size: 0.9rem;
                    color: #666;
                }}
                
                .insights-section {{
                    background: linear-gradient(135deg, #e8f5e8 0%, #d4f1d4 100%);
                    padding: 40px;
                    margin: 0;
                }}
                
                .insights-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 25px;
                    margin-top: 25px;
                }}
                
                .insight-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    border-left: 5px solid #25D366;
                }}
                
                .insight-title {{
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 10px;
                    font-size: 1.1rem;
                }}
                
                .insight-content {{
                    color: #555;
                    line-height: 1.6;
                }}
                
                .chart-container {{
                    margin: 30px 0;
                    background: white;
                    border-radius: 12px;
                    padding: 20px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                
                .wordcloud-container {{
                    text-align: center;
                    padding: 30px;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    margin: 30px 0;
                }}
                
                .wordcloud-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                
                .footer {{
                    background: #2c3e50;
                    color: white;
                    text-align: center;
                    padding: 30px;
                    font-size: 0.9rem;
                }}
                
                .footer a {{
                    color: #25D366;
                    text-decoration: none;
                }}
                
                .timeline-item {{
                    background: #f8f9fa;
                    padding: 15px;
                    margin: 10px 0;
                    border-left: 4px solid #25D366;
                    border-radius: 0 8px 8px 0;
                }}
                
                @media (max-width: 768px) {{
                    .header h1 {{ font-size: 2rem; }}
                    .overview-grid {{ grid-template-columns: 1fr; padding: 20px; }}
                    .section {{ padding: 20px; }}
                    .participants-grid {{ grid-template-columns: 1fr; }}
                }}
                
                .emoji-large {{ font-size: 1.5rem; margin-right: 8px; }}
                .progress-bar {{
                    width: 100%;
                    height: 8px;
                    background: #eee;
                    border-radius: 4px;
                    overflow: hidden;
                    margin: 5px 0;
                }}
                .progress-fill {{
                    height: 100%;
                    background: linear-gradient(90deg, #25D366, #128C7E);
                    transition: width 0.3s ease;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header class="header">
                    <h1>📱 Enhanced WhatsApp Analytics</h1>
                    <p class="subtitle">Comprehensive Chat Analysis & Insights</p>
                    <p style="margin-top: 15px; font-size: 1.1rem;">
                        📅 {date_range['start_formatted']} - {date_range['end_formatted']}
                    </p>
                </header>
                
                <div class="overview-grid">
                    <div class="stat-card messages">
                        <div class="stat-number" style="color: #25D366;">{analytics['total_messages']:,}</div>
                        <div class="stat-label">💬 Total Messages</div>
                    </div>
                    <div class="stat-card participants">
                        <div class="stat-number" style="color: #128C7E;">{analytics['total_participants']}</div>
                        <div class="stat-label">👥 Active Participants</div>
                    </div>
                    <div class="stat-card days">
                        <div class="stat-number" style="color: #34495e;">{date_range['duration_days']}</div>
                        <div class="stat-label">📅 Days Analyzed</div>
                    </div>
                    <div class="stat-card media">
                        <div class="stat-number" style="color: #e74c3c;">{analytics['content_analysis']['total_media']:,}</div>
                        <div class="stat-label">📱 Media Shared</div>
                    </div>
                    <div class="stat-card sentiment">
                        <div class="stat-number" style="color: #f39c12;">{'😊' if sentiment['overall_sentiment'] > 0.1 else '😐' if sentiment['overall_sentiment'] > -0.1 else '😔'}</div>
                        <div class="stat-label">🎭 Overall Sentiment</div>
                    </div>
                    <div class="stat-card peak">
                        <div class="stat-number" style="color: #9b59b6;">{peak_activity['peak_hour']}:00</div>
                        <div class="stat-label">🔥 Peak Hour</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2><i class="fas fa-chart-line"></i>Interactive Analytics Dashboard</h2>
                    <div class="chart-container">
                        <div id="main-dashboard" style="width:100%;height:3200px;"></div>
                    </div>
                </div>
                
                <div class="section">
                    <h2><i class="fas fa-project-diagram"></i>Relationship Network</h2>
                    <div class="chart-container">
                        <div id="network-chart" style="width:100%;height:600px;"></div>
                    </div>
                </div>
                
                <div class="section">
                    <h2><i class="fas fa-users"></i>Participant Deep Dive</h2>
                    <div class="participants-grid">
        """
        
        # Add participant cards with detailed stats
        sorted_participants = sorted(analytics['participant_stats'].items(), 
                                   key=lambda x: x[1]['total_messages'], reverse=True)
        
        for i, (participant, stats) in enumerate(sorted_participants[:12], 1):
            engagement_score = (stats['total_messages'] * 0.3 + 
                              stats['conversation_starters'] * 0.4 + 
                              stats['questions_asked'] * 0.2 + 
                              stats['total_emojis'] * 0.1)
            
            sentiment_emoji = '😊' if analytics['sentiment_analysis']['sentiment_by_participant'].get(participant, 0) > 0.1 else '😐' if analytics['sentiment_analysis']['sentiment_by_participant'].get(participant, 0) > -0.1 else '😔'
            
            html_content += f"""
                        <div class="participant-card">
                            <div class="participant-name">#{i} {participant}</div>
                            <div class="participant-stats">
                                <div>💬 Messages: <strong>{stats['total_messages']:,}</strong></div>
                                <div>📝 Avg Words: <strong>{stats['avg_words_per_message']:.1f}</strong></div>
                                <div>🚀 Conversations Started: <strong>{stats['conversation_starters']}</strong></div>
                                <div>❓ Questions Asked: <strong>{stats['questions_asked']}</strong></div>
                                <div>😀 Emojis Used: <strong>{stats['total_emojis']}</strong></div>
                                <div>📱 Media Shared: <strong>{stats['media_messages']}</strong></div>
                                <div>🎭 Sentiment: <strong>{sentiment_emoji}</strong></div>
                                <div>⭐ Engagement: <strong>{engagement_score:.1f}</strong></div>
                                <div>🕐 Most Active: <strong>{stats['most_active_hour']}:00</strong></div>
                                <div>📅 Favorite Day: <strong>{stats['most_active_day']}</strong></div>
                            </div>
                        </div>
            """
        
        html_content += f"""
                    </div>
                </div>
                
                <div class="wordcloud-container">
                    <h2 style="margin-bottom: 20px;"><i class="fas fa-cloud"></i>Enhanced Word Cloud</h2>
                    <img src="enhanced_wordcloud.png" alt="Enhanced Word Cloud">
                </div>
                
                <div class="insights-section">
                    <h2 style="color: #2c3e50;"><i class="fas fa-lightbulb"></i>🎯 Advanced Insights & Discoveries</h2>
                    <div class="insights-grid">
                        <div class="insight-card">
                            <div class="insight-title">👑 Group Dynamics</div>
                            <div class="insight-content">
                                <strong>{peak_activity['busiest_participant']}</strong> is the most active participant with the highest message count. 
                                <strong>{peak_activity['most_verbose_participant']}</strong> writes the longest messages on average.
                                The group shows {'high' if analytics['total_messages']/date_range['duration_days'] > 100 else 'moderate' if analytics['total_messages']/date_range['duration_days'] > 50 else 'low'} 
                                activity with {analytics['total_messages']/date_range['duration_days']:.1f} messages per day.
                            </div>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-title">🕐 Activity Patterns</div>
                            <div class="insight-content">
                                Peak activity occurs at <strong>{peak_activity['peak_hour']}:00</strong> with {peak_activity['peak_hour_messages']} messages.
                                <strong>{peak_activity['peak_day']}</strong> is the most active day with {peak_activity['peak_day_messages']} messages.
                                The group is most active during <strong>{max(analytics['time_period_activity'], key=analytics['time_period_activity'].get)}</strong>.
                            </div>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-title">💭 Communication Style</div>
                            <div class="insight-content">
                                Average message length is <strong>{analytics['content_analysis']['avg_message_length']:.0f}</strong> characters.
                                <strong>{analytics['content_analysis']['total_questions']:,}</strong> questions were asked, showing active engagement.
                                <strong>{peak_activity['emoji_king']}</strong> is the emoji champion with the most emoji usage.
                            </div>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-title">🎭 Sentiment Analysis</div>
                            <div class="insight-content">
                                Overall group sentiment is <strong>{'Positive 😊' if sentiment['overall_sentiment'] > 0.1 else 'Neutral 😐' if sentiment['overall_sentiment'] > -0.1 else 'Negative 😔'}</strong> 
                                ({sentiment['overall_sentiment']:.3f}).
                                <strong>{sentiment['sentiment_distribution']['positive']}</strong> positive messages, 
                                <strong>{sentiment['sentiment_distribution']['neutral']}</strong> neutral, and 
                                <strong>{sentiment['sentiment_distribution']['negative']}</strong> negative messages were detected.
                            </div>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-title">🚀 Conversation Initiators</div>
                            <div class="insight-content">
                                Top conversation starters: <strong>{', '.join(list(analytics['conversation_patterns']['top_initiators'].keys())[:3])}</strong>.
                                <strong>{peak_activity['question_master']}</strong> asks the most questions, driving discussions.
                                Average response time is <strong>{list(analytics['conversation_patterns']['avg_response_times'].values())[0]:.1f}</strong> minutes.
                            </div>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-title">📊 Content Breakdown</div>
                            <div class="insight-content">
                                <strong>{analytics['content_analysis']['total_media']:,}</strong> media files shared ({analytics['content_analysis']['total_media']/analytics['total_messages']*100:.1f}% of messages).
                                <strong>{analytics['content_analysis']['total_urls']:,}</strong> URLs shared, showing information exchange.
                                <strong>{analytics['content_analysis']['total_deleted']:,}</strong> messages were deleted.
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2><i class="fas fa-trophy"></i>🏆 Hall of Fame</h2>
                    <div class="insights-grid">
                        <div class="insight-card">
                            <div class="insight-title">🥇 Most Active</div>
                            <div class="insight-content">
                                <div class="emoji-large">👑</div>
                                <strong>{peak_activity['busiest_participant']}</strong><br>
                                {analytics['participant_stats'][peak_activity['busiest_participant']]['total_messages']:,} messages
                            </div>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-title">🥈 Most Verbose</div>
                            <div class="insight-content">
                                <div class="emoji-large">📝</div>
                                <strong>{peak_activity['most_verbose_participant']}</strong><br>
                                {analytics['participant_stats'][peak_activity['most_verbose_participant']]['avg_words_per_message']:.1f} words/message
                            </div>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-title">🥉 Emoji King</div>
                            <div class="insight-content">
                                <div class="emoji-large">😀</div>
                                <strong>{peak_activity['emoji_king']}</strong><br>
                                {analytics['participant_stats'][peak_activity['emoji_king']]['total_emojis']:,} emojis used
                            </div>
                        </div>
                        
                        <div class="insight-card">
                            <div class="insight-title">🏅 Question Master</div>
                            <div class="insight-content">
                                <div class="emoji-large">❓</div>
                                <strong>{peak_activity['question_master']}</strong><br>
                                {analytics['participant_stats'][peak_activity['question_master']]['questions_asked']:,} questions asked
                            </div>
                        </div>
                    </div>
                </div>
                
                <footer class="footer">
                    <p>📊 Enhanced WhatsApp Analytics Report | Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    <p>🚀 Powered by Advanced Analytics Engine | <a href="#">View Source Code</a></p>
                </footer>
            </div>
            
            <script>
                // Main Dashboard
                var mainDashboardData = {main_fig.to_json()};
                Plotly.newPlot('main-dashboard', mainDashboardData.data, mainDashboardData.layout, {{responsive: true}});
                
                // Network Chart
                var networkData = {network_fig.to_json()};
                Plotly.newPlot('network-chart', networkData.data, networkData.layout, {{responsive: true}});
                
                // Add smooth scroll behavior
                document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                    anchor.addEventListener('click', function (e) {{
                        e.preventDefault();
                        document.querySelector(this.getAttribute('href')).scrollIntoView({{
                            behavior: 'smooth'
                        }});
                    }});
                }});
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def _save_enhanced_analytics(self, analytics, output_dir):
        """Save enhanced analytics to JSON with proper serialization"""
        import json
        
        def serialize_data(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return obj
        
        # Deep clean analytics data
        cleaned_analytics = json.loads(json.dumps(analytics, default=serialize_data))
        
        json_path = os.path.join(output_dir, 'enhanced_analytics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_analytics, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Enhanced WhatsApp Chat Analytics Tool')
    parser.add_argument('file_path', help='Path to WhatsApp chat export file')
    parser.add_argument('--output', '-o', default='enhanced_output', help='Output directory (default: enhanced_output)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File '{args.file_path}' not found.")
        return
    
    print("🚀 Starting Enhanced WhatsApp Chat Analysis...")
    print(f"📁 Input file: {args.file_path}")
    print(f"📁 Output directory: {args.output}")
    
    try:
        analyzer = EnhancedWhatsAppAnalyzer(args.file_path)
        report_path, analytics = analyzer.generate_enhanced_report(args.output)
        
        print(f"\n✅ Enhanced Analysis Complete!")
        print(f"📊 Processed {analytics['total_messages']:,} messages")
        print(f"👥 Found {analytics['total_participants']} participants")
        print(f"📅 Date range: {analytics['date_range']['duration_days']} days")
        print(f"🎭 Overall sentiment: {'Positive 😊' if analytics['sentiment_analysis']['overall_sentiment'] > 0.1 else 'Neutral 😐' if analytics['sentiment_analysis']['overall_sentiment'] > -0.1 else 'Negative 😔'}")
        print(f"📝 Enhanced report saved to: {report_path}")
        print(f"\n🌐 Open {report_path} in your browser to explore the enhanced analytics!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("💡 Please check your file format and try again.")

if __name__ == "__main__":
    main()
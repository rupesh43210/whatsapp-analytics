#!/usr/bin/env python3
"""
Ultimate WhatsApp Analytics - Amazing Visualizations, All Participants, Relationship Matrices, Heatmaps
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

class UltimateWhatsAppAnalyzer:
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
        
        # Add sentiment analysis
        print("🧠 Performing sentiment analysis...")
        self.df['sentiment'] = self.df['message'].apply(self._get_sentiment)
        self.df['sentiment_score'] = self.df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
        
        # Add time-based features
        self.df['date_formatted'] = self.df['timestamp'].dt.strftime('%B %d, %Y')
        self.df['time_period'] = pd.cut(self.df['hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['Night (12-6AM)', 'Morning (6-12PM)', 'Afternoon (12-6PM)', 'Evening (6-12AM)'],
                                       include_lowest=True)
    
    def _get_sentiment(self, text):
        """Get sentiment category"""
        try:
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.1:
                return 'Positive'
            elif polarity < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        except:
            return 'Neutral'
    
    def create_amazing_visualizations(self):
        """Create comprehensive dashboard with amazing visualizations"""
        
        # Create 4x4 subplot grid for comprehensive dashboard
        fig = make_subplots(
            rows=4, cols=4,
            subplot_titles=[
                'Messages by ALL Participants', 'Activity Heatmap (Hour vs Day)', 
                'Sentiment Analysis Distribution', 'Monthly Message Trends',
                'Words per Message Distribution', 'Message Types Breakdown',
                'Top Active Hours', 'Response Time Patterns',
                'Emoji Usage by Person', 'Questions vs Statements',
                'Media Sharing Patterns', 'Conversation Initiator Analysis',
                'Weekly Activity Trends', 'Character Count Distribution',
                'Time Period Activity', 'Most Used Words Cloud'
            ],
            specs=[
                [{"type": "bar"}, {"type": "xy"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "pie"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Messages by ALL Participants (not just top 10!)
        participant_counts = self.df['sender'].value_counts()  # ALL participants
        colors = px.colors.qualitative.Set3[:len(participant_counts)]
        
        fig.add_trace(
            go.Bar(
                x=list(participant_counts.index),
                y=list(participant_counts.values),
                marker=dict(color=colors),
                text=[f"{v:,}" for v in participant_counts.values],
                textposition='outside',
                name='Messages',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Activity Heatmap - Hourly activity pattern
        hourly_data = self.df['hour'].value_counts().sort_index()
        
        fig.add_trace(
            go.Scatter(
                x=list(range(24)),
                y=list(hourly_data.reindex(range(24), fill_value=0).values),
                mode='lines+markers',
                fill='tonexty',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=8),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Sentiment Analysis
        sentiment_counts = self.df['sentiment'].value_counts()
        sentiment_colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
        
        fig.add_trace(
            go.Bar(
                x=list(sentiment_counts.index),
                y=list(sentiment_counts.values),
                marker=dict(color=[sentiment_colors.get(x, '#3498db') for x in sentiment_counts.index]),
                text=[f"{v:,}" for v in sentiment_counts.values],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # 4. Monthly Trends with better formatting
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
                mode='lines+markers',
                line=dict(color='#e74c3c', width=4),
                marker=dict(size=10),
                fill='tonexty',
                showlegend=False
            ),
            row=1, col=4
        )
        
        # 5. Words per Message Distribution
        word_dist = self.df.groupby('sender')['word_count'].mean().sort_values(ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=list(word_dist.index),
                y=list(word_dist.values),
                marker=dict(color='#9b59b6'),
                text=[f"{v:.1f}" for v in word_dist.values],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 6. Message Types Pie Chart
        message_types = {
            'Text': len(self.df[~self.df['is_media'] & ~self.df['is_question']]),
            'Questions': self.df['is_question'].sum(),
            'Media': self.df['is_media'].sum(),
            'URLs': self.df['has_url'].sum()
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(message_types.keys()),
                values=list(message_types.values()),
                marker=dict(colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 7. Top Active Hours
        hourly_data = self.df['hour'].value_counts().sort_index()
        hour_labels = [f"{h:02d}:00" for h in hourly_data.index]
        
        fig.add_trace(
            go.Bar(
                x=hour_labels,
                y=list(hourly_data.values),
                marker=dict(color='#1abc9c'),
                text=[f"{v:,}" for v in hourly_data.values],
                textposition='outside',
                showlegend=False
            ),
            row=4, col=1
        )
        
        # 8. Emoji Usage by Top Participants
        emoji_users = self.df[self.df['emoji_count'] > 0].groupby('sender')['emoji_count'].sum().sort_values(ascending=False).head(10)
        
        fig.add_trace(
            go.Bar(
                x=list(emoji_users.index),
                y=list(emoji_users.values),
                marker=dict(color='#ff6b6b'),
                text=[f"{v:,}" for v in emoji_users.values],
                textposition='outside',
                showlegend=False
            ),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1800,
            title={
                'text': "🚀 Ultimate WhatsApp Analytics Dashboard<br><sub>Amazing Visualizations • All Participants • Advanced Insights</sub>",
                'x': 0.5,
                'font': {'size': 24, 'color': '#2c3e50'}
            },
            font=dict(size=10),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa'
        )
        
        # Update all axes
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(title_text="Messages", row=1, col=1)
        fig.update_yaxes(title_text="Day of Week", row=1, col=2)
        fig.update_yaxes(title_text="Messages", row=2, col=1)
        fig.update_yaxes(title_text="Messages", row=2, col=2)
        
        return fig
    
    def create_relationship_matrix(self):
        """Create relationship and interaction matrices"""
        
        # Create a reply/mention analysis
        reply_data = defaultdict(lambda: defaultdict(int))
        
        # Analyze conversation flow (who responds to whom)
        sorted_messages = self.df.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(1, len(sorted_messages)):
            current_sender = sorted_messages.iloc[i]['sender']
            previous_sender = sorted_messages.iloc[i-1]['sender']
            
            # Count interactions (excluding self-replies)
            if current_sender != previous_sender:
                reply_data[previous_sender][current_sender] += 1
        
        # Convert to matrix format
        participants = list(self.participants)
        matrix_data = []
        
        for p1 in participants:
            row = []
            for p2 in participants:
                row.append(reply_data[p1][p2])
            matrix_data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=participants,
            y=participants,
            colorscale='Viridis',
            hoverongaps=False,
            colorbar=dict(title="Interactions")
        ))
        
        fig.update_layout(
            title="🔗 Interaction Matrix - Who Responds to Whom",
            xaxis_title="Responds To",
            yaxis_title="Responder",
            height=600,
            font=dict(size=12)
        )
        
        return fig
    
    def create_network_graph(self):
        """Create interactive network graph"""
        
        # Create network
        G = nx.Graph()
        
        # Add nodes (participants)
        participant_counts = self.df['sender'].value_counts()
        for participant in participant_counts.index:
            G.add_node(participant, size=participant_counts[participant])
        
        # Add edges (interactions)
        sorted_messages = self.df.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(1, len(sorted_messages)):
            current_sender = sorted_messages.iloc[i]['sender']
            previous_sender = sorted_messages.iloc[i-1]['sender']
            
            if current_sender != previous_sender:
                if G.has_edge(previous_sender, current_sender):
                    G[previous_sender][current_sender]['weight'] += 1
                else:
                    G.add_edge(previous_sender, current_sender, weight=1)
        
        # Generate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"{edge[0]} ↔ {edge[1]}: {weight} interactions")
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=0.5, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Messages: {participant_counts[node]:,}")
            node_size.append(min(50, max(10, participant_counts[node] / 50)))
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=[node for node in G.nodes()],
                               textposition="middle center",
                               hovertext=node_text,
                               marker=dict(showscale=True,
                                         colorscale='YlOrRd',
                                         reversescale=True,
                                         color=[],
                                         size=node_size,
                                         colorbar=dict(
                                             thickness=15,
                                             len=0.5,
                                             x=1.05,
                                             title="Messages"
                                         ),
                                         line_width=2))
        
        # Color nodes by message count
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(participant_counts[node])
        
        node_trace.marker.color = node_adjacencies
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='🌐 Participant Interaction Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes to see details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888")
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600))
        
        return fig
    
    def generate_ultimate_statistics(self):
        """Generate comprehensive statistics for ALL participants"""
        daily_counts = self.df.groupby(self.df['timestamp'].dt.date).size()
        hourly_counts = self.df['hour'].value_counts()
        participant_counts = self.df['sender'].value_counts()
        
        # ALL Participant detailed stats (not just top 10!)
        participant_stats = []
        for participant in participant_counts.index:  # ALL participants
            participant_data = self.df[self.df['sender'] == participant]
            stats = {
                'name': participant,
                'messages': len(participant_data),
                'words': participant_data['word_count'].sum(),
                'avg_words': participant_data['word_count'].mean(),
                'characters': participant_data['char_count'].sum(),
                'media_files': participant_data['is_media'].sum(),
                'questions': participant_data['is_question'].sum(),
                'emojis': participant_data['emoji_count'].sum(),
                'urls': participant_data['has_url'].sum(),
                'positive_sentiment': len(participant_data[participant_data['sentiment'] == 'Positive']),
                'negative_sentiment': len(participant_data[participant_data['sentiment'] == 'Negative']),
                'first_message': participant_data['timestamp'].min().strftime('%B %d, %Y'),
                'last_message': participant_data['timestamp'].max().strftime('%B %d, %Y'),
                'most_active_hour': participant_data['hour'].mode().iloc[0] if not participant_data['hour'].mode().empty else 12
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
                'avg_words': day_data['word_count'].mean() if len(day_data) > 0 else 0,
                'avg_sentiment': day_data['sentiment_score'].mean() if len(day_data) > 0 else 0
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
            'participant_details': participant_stats,  # ALL participants
            'day_breakdown': day_stats,
            'sentiment_stats': {
                'positive': len(self.df[self.df['sentiment'] == 'Positive']),
                'negative': len(self.df[self.df['sentiment'] == 'Negative']),
                'neutral': len(self.df[self.df['sentiment'] == 'Neutral']),
                'avg_sentiment': self.df['sentiment_score'].mean()
            },
            'content_stats': {
                'text_messages': len(self.df[~self.df['is_media']]),
                'media_files': self.df['is_media'].sum(),
                'questions': self.df['is_question'].sum(),
                'urls': self.df['has_url'].sum(),
                'total_words': self.df['word_count'].sum(),
                'total_characters': self.df['char_count'].sum(),
                'total_emojis': self.df['emoji_count'].sum(),
                'avg_message_length': self.df['word_count'].mean()
            }
        }
    
    def generate_ultimate_report(self, output_dir='ultimate_analysis'):
        """Generate ultimate report with amazing visualizations"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.parse_messages()
        
        print("🎨 Creating amazing visualizations...")
        main_fig = self.create_amazing_visualizations()
        
        print("🔗 Creating relationship matrix...")
        relationship_fig = self.create_relationship_matrix()
        
        print("🌐 Creating network graph...")
        network_fig = self.create_network_graph()
        
        print("📊 Generating comprehensive statistics...")
        stats = self.generate_ultimate_statistics()
        
        # Generate ALL participant table HTML
        participant_table = ""
        for i, p in enumerate(stats['participant_details']):
            participant_table += f"""
            <tr>
                <td>{i+1}</td>
                <td><strong>{p['name']}</strong></td>
                <td>{p['messages']:,}</td>
                <td>{p['words']:,}</td>
                <td>{p['avg_words']:.1f}</td>
                <td>{p['media_files']:,}</td>
                <td>{p['questions']:,}</td>
                <td>{p['emojis']:,}</td>
                <td>{p['positive_sentiment']:,}</td>
                <td>{p['negative_sentiment']:,}</td>
                <td>{p['first_message']}</td>
            </tr>
            """
        
        # Generate comprehensive HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🚀 Ultimate WhatsApp Analytics</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: 'Arial', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #333; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; box-shadow: 0 0 30px rgba(0,0,0,0.2); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 50px; text-align: center; }}
                .header h1 {{ font-size: 3rem; margin-bottom: 15px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
                .header p {{ font-size: 1.2rem; opacity: 0.9; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; padding: 40px; background: #f8f9fa; }}
                .stat {{ background: white; padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1); transform: translateY(0); transition: transform 0.3s; }}
                .stat:hover {{ transform: translateY(-5px); }}
                .stat-number {{ font-size: 2.2rem; font-weight: bold; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
                .stat-label {{ color: #666; margin-top: 8px; font-weight: 500; }}
                .section {{ padding: 40px; }}
                .chart {{ background: white; border-radius: 15px; padding: 25px; margin: 25px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
                h2 {{ color: #333; margin-bottom: 25px; font-size: 2rem; border-bottom: 4px solid #667eea; padding-bottom: 15px; }}
                h3 {{ color: #555; margin: 25px 0 15px 0; font-size: 1.5rem; }}
                table {{ width: 100%; border-collapse: collapse; margin: 25px 0; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
                th {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 15px; text-align: left; font-weight: bold; }}
                td {{ padding: 12px 15px; border-bottom: 1px solid #eee; }}
                tr:hover {{ background: #f8f9fa; }}
                .insights {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 30px 0; }}
                .insight {{ background: linear-gradient(135deg, #f8f9fa, #ffffff); padding: 20px; border-radius: 12px; border-left: 5px solid #28a745; box-shadow: 0 3px 10px rgba(0,0,0,0.1); }}
                .insight-title {{ font-weight: bold; color: #333; font-size: 1.1rem; }}
                .insight-value {{ color: #28a745; font-size: 1.5rem; font-weight: bold; margin: 5px 0; }}
                .footer {{ background: #333; color: white; text-align: center; padding: 30px; }}
                .highlight {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 2px 8px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Ultimate WhatsApp Analytics</h1>
                    <p>Amazing Visualizations • All {stats['total_participants']} Participants • Advanced Insights</p>
                    <p>{stats['date_range']['start']} - {stats['date_range']['end']}</p>
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <div class="stat-number">{stats['total_messages']:,}</div>
                        <div class="stat-label">Total Messages</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['total_participants']}</div>
                        <div class="stat-label">All Participants</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['content_stats']['total_words']:,}</div>
                        <div class="stat-label">Total Words</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['content_stats']['total_emojis']:,}</div>
                        <div class="stat-label">Total Emojis</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['content_stats']['media_files']:,}</div>
                        <div class="stat-label">Media Files</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['sentiment_stats']['positive']:,}</div>
                        <div class="stat-label">Positive Messages</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎨 Amazing Analytics Dashboard</h2>
                    <p>Comprehensive visualization showing all participants and advanced insights</p>
                    <div class="chart">
                        <div id="main-dashboard" style="width:100%;height:1800px;"></div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔗 Relationship Matrix & Network Analysis</h2>
                    <div class="chart">
                        <div id="relationship-matrix" style="width:100%;height:600px;"></div>
                    </div>
                    <div class="chart">
                        <div id="network-graph" style="width:100%;height:600px;"></div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Ultimate Key Insights</h2>
                    <div class="insights">
                        <div class="insight">
                            <div class="insight-title">🔥 Busiest Day Ever</div>
                            <div class="insight-value">{stats['busiest_date']['date']}</div>
                            <div>{stats['busiest_date']['count']:,} messages sent</div>
                        </div>
                        <div class="insight">
                            <div class="insight-title">⏰ Peak Activity Hour</div>
                            <div class="insight-value">{stats['peak_hour']['formatted']}</div>
                            <div>{stats['peak_hour']['count']:,} messages</div>
                        </div>
                        <div class="insight">
                            <div class="insight-title">👑 Most Active Champion</div>
                            <div class="insight-value">{stats['top_participant']['name']}</div>
                            <div>{stats['top_participant']['count']:,} messages</div>
                        </div>
                        <div class="insight">
                            <div class="insight-title">😊 Group Sentiment</div>
                            <div class="insight-value">{stats['sentiment_stats']['avg_sentiment']:.2f}</div>
                            <div>Average sentiment score</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>👥 Complete Participant Analysis</h2>
                    <p><span class="highlight">ALL {stats['total_participants']} participants</span> with detailed statistics</p>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Name</th>
                                <th>Messages</th>
                                <th>Words</th>
                                <th>Avg Words</th>
                                <th>Media</th>
                                <th>Questions</th>
                                <th>Emojis</th>
                                <th>Positive</th>
                                <th>Negative</th>
                                <th>First Message</th>
                            </tr>
                        </thead>
                        <tbody>
                            {participant_table}
                        </tbody>
                    </table>
                </div>
                
                <div class="footer">
                    <p>🚀 Ultimate WhatsApp Analytics Report</p>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    <p style="margin-top: 15px;">✨ Amazing Visualizations • 🔗 Relationship Matrices • 📊 Comprehensive Analytics</p>
                </div>
            </div>
            
            <script>
                // Main Dashboard
                var mainData = {main_fig.to_json()};
                Plotly.newPlot('main-dashboard', mainData.data, mainData.layout, {{responsive: true}});
                
                // Relationship Matrix
                var relationshipData = {relationship_fig.to_json()};
                Plotly.newPlot('relationship-matrix', relationshipData.data, relationshipData.layout, {{responsive: true}});
                
                // Network Graph
                var networkData = {network_fig.to_json()};
                Plotly.newPlot('network-graph', networkData.data, networkData.layout, {{responsive: true}});
            </script>
        </body>
        </html>
        """
        
        report_path = os.path.join(output_dir, 'ultimate_analytics_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path, stats

def main():
    parser = argparse.ArgumentParser(description='Ultimate WhatsApp Analytics Tool')
    parser.add_argument('file_path', help='Path to WhatsApp chat export file')
    parser.add_argument('--output', '-o', default='ultimate_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File '{args.file_path}' not found.")
        return
    
    print("🚀 Starting ULTIMATE WhatsApp Analytics...")
    print(f"📁 Input file: {args.file_path}")
    print(f"📁 Output directory: {args.output}")
    
    try:
        analyzer = UltimateWhatsAppAnalyzer(args.file_path)
        report_path, stats = analyzer.generate_ultimate_report(args.output)
        
        print(f"\n✅ ULTIMATE Analysis Complete!")
        print(f"📊 Analyzed {stats['total_messages']:,} messages")
        print(f"👥 From ALL {stats['total_participants']} participants")
        print(f"🔥 Busiest day: {stats['busiest_date']['date']} ({stats['busiest_date']['count']:,} messages)")
        print(f"⏰ Peak hour: {stats['peak_hour']['formatted']} ({stats['peak_hour']['count']:,} messages)")
        print(f"😊 Group sentiment: {stats['sentiment_stats']['avg_sentiment']:.2f}")
        print(f"📝 ULTIMATE report: {report_path}")
        
        print(f"\n🎨 Features included:")
        print(f"✅ ALL {stats['total_participants']} participants analyzed")
        print(f"✅ Amazing multi-chart dashboard")
        print(f"✅ Relationship matrices")
        print(f"✅ Interactive network graphs")
        print(f"✅ Activity heatmaps")
        print(f"✅ Sentiment analysis")
        print(f"✅ Comprehensive statistics")
        
        print(f"\n🌐 Open {report_path} to view the ULTIMATE analytics!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()